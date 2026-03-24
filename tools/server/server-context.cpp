#include "server-context.h"
#include "server-bash-tool.h"
#include "server-common.h"
#include "server-http.h"
#include "server-openclaw-fabric.h"
#include "server-task.h"
#include "server-queue.h"

#include "chat.h"
#include "base64.hpp"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "ggml.h"
#include "llama.h"
#include "llama-cpp.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json_fwd.hpp>
#include <nlohmann/json.hpp>

#include "../../src/llama-context.h"
#include "../../src/llama-hard-memory.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;
constexpr uint64_t VICUNA_DMN_IDLE_TICK_MIN_INTERVAL_US = 5ULL * 1000ULL * 1000ULL;
constexpr size_t VICUNA_TELEGRAM_DIALOGUE_MAX_TURNS_PER_SCOPE = 16;
constexpr int32_t VICUNA_TELEGRAM_DIALOGUE_DEFAULT_TURN_LIMIT = 6;
constexpr const char * VICUNA_TELEGRAM_DIALOGUE_SCOPE_BROADCAST = "__broadcast__";

struct self_state_token_view {
    llama_tokens owned_tokens;
    const llama_token * data = nullptr;
    size_t size = 0;
};

static self_state_token_view make_self_state_token_view(
        const server_tokens & tokens,
        size_t trim_prefix_tokens = 0) {
    self_state_token_view view;
    if (!tokens.has_mtmd) {
        const llama_tokens & text_tokens = tokens.get_text_tokens();
        const size_t offset = std::min(trim_prefix_tokens, text_tokens.size());
        view.data = offset >= text_tokens.size() ? nullptr : text_tokens.data() + offset;
        view.size = text_tokens.size() - offset;
        return view;
    }

    // Active-loop/self-state ingestion is text-native. For multimodal prompts,
    // preserve the textual thread while excluding media placeholders.
    view.owned_tokens.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        const llama_token token = tokens[i];
        if (token != LLAMA_TOKEN_NULL) {
            view.owned_tokens.push_back(token);
        }
    }
    const size_t offset = std::min(trim_prefix_tokens, view.owned_tokens.size());
    view.data = offset >= view.owned_tokens.size() ? nullptr : view.owned_tokens.data() + offset;
    view.size = view.owned_tokens.size() - offset;
    return view;
}

static self_state_token_view make_self_state_token_view(
        const server_tokens & tokens,
        const llama_tokens * foreground_tokens,
        size_t trim_prefix_tokens = 0) {
    if (foreground_tokens && !foreground_tokens->empty()) {
        self_state_token_view view;
        view.owned_tokens = *foreground_tokens;
        view.data = view.owned_tokens.data();
        view.size = view.owned_tokens.size();
        return view;
    }

    return make_self_state_token_view(tokens, trim_prefix_tokens);
}

static bool admit_runtime_emit_tokens(
        llama_context * ctx,
        const llama_token * tokens,
        size_t n_tokens,
        int32_t loop_origin,
        int32_t phase,
        int32_t source_id,
        int32_t plan_id,
        uint32_t extra_flags = 0,
        int32_t artifact_kind = LLAMA_SELF_COG_ARTIFACT_EXTERNAL_EVENT) {
    if (!ctx || !tokens || n_tokens == 0) {
        return false;
    }

    (void) llama_self_state_set_channel_state(ctx, LLAMA_SELF_STATE_CHANNEL_ACTIVE);

    llama_self_state_event event = {
        /*.tokens =*/ tokens,
        /*.n_tokens =*/ n_tokens,
        /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ (uint32_t) (LLAMA_SELF_STATE_EVENT_ADMITTED | extra_flags),
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
        /*.artifact_kind =*/ artifact_kind,
        /*.loop_origin =*/ loop_origin,
        /*.phase =*/ phase,
        /*.source_id =*/ source_id,
        /*.plan_id =*/ plan_id,
    };
    llama_self_state_feature_vector pre = {};
    llama_self_state_feature_vector post = {};
    const bool admitted =
            llama_self_state_build_prewrite_features(ctx, &event, &pre) == 0 &&
            llama_self_state_apply_prewrite(ctx, &event, &pre) == 0 &&
            llama_self_state_build_postwrite_features(ctx, &event, &post) == 0 &&
            llama_self_state_apply_postwrite(ctx, &event, &post) == 0;
    if (!admitted) {
        return false;
    }
    return true;
}

static bool admit_runtime_emit_text(
        llama_context * ctx,
        const std::string & text,
        int32_t loop_origin,
        int32_t phase,
        int32_t source_id,
        int32_t plan_id,
        uint32_t extra_flags = 0,
        int32_t artifact_kind = LLAMA_SELF_COG_ARTIFACT_EXTERNAL_EVENT) {
    if (!ctx) {
        return false;
    }

    const std::string trimmed = string_strip(text);
    if (trimmed.empty()) {
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = model ? llama_model_get_vocab(model) : nullptr;
    if (!vocab) {
        return false;
    }

    const llama_tokens tokens = common_tokenize(vocab, trimmed, true, true);
    return admit_runtime_emit_tokens(
            ctx,
            tokens.empty() ? nullptr : tokens.data(),
            tokens.size(),
            loop_origin,
            phase,
            source_id,
            plan_id,
            extra_flags,
            artifact_kind);
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_REPLAYING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot {
    int id;

    // TODO: change to unique_ptrs for consistency:
    llama_context * ctx = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;

    common_speculative * spec = nullptr;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<server_task> task;
    std::unique_ptr<server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    // idx of draft tokens in the main batch
    // non-empty if we went to evaluate draft tokens
    // ref: https://github.com/ggml-org/llama.cpp/pull/17808
    std::vector<int32_t> i_batch_dft;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;
    std::unique_ptr<server_tokens> replay_tokens;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.empty());

        const size_t cur_size = llama_state_seq_get_size_ext(ctx, id, 0);

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx, cur->data.data(), cur_size, id, 0);
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        llama_memory_seq_rm(llama_get_memory(ctx), id, -1, -1);
        prompt.tokens.clear();
        replay_tokens.reset();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token  sampled; // in speculative mode, this is the last accepted token
    llama_tokens drafted;

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing; // ms
    double t_token_generation;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        drafted.clear();
        i_batch_dft.clear();
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;

        task_prev = std::move(task);
        task.reset();
        replay_tokens.reset();

        llama_set_sampler(ctx, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_INF(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool is_replaying_prompt() const {
        return replay_tokens != nullptr;
    }

    const server_tokens & input_tokens_ref() const {
        return replay_tokens ? *replay_tokens : task->tokens;
    }

    int input_tokens_count() const {
        return replay_tokens ? (int) replay_tokens->size() : task->n_tokens();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx) && llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        int n_draft_max = task->params.speculative.n_max;

        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        if (n_draft_max < task->params.speculative.n_min) {
            SLT_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, task->params.speculative.n_min);
            n_draft_max = 0;
        }

        return n_draft_max;
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_CNT(*this,
                    "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total
            );
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        llama_memory_seq_rm(llama_get_memory(ctx), other.id,     -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx), id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

enum vicuna_external_work_kind {
    VICUNA_EXTERNAL_WORK_BASH = 0,
    VICUNA_EXTERNAL_WORK_HARD_MEMORY = 1,
    VICUNA_EXTERNAL_WORK_CODEX = 2,
};

struct vicuna_external_work_item {
    int32_t command_id = -1;
    int32_t origin = -1;
    int32_t tool_kind = LLAMA_TOOL_KIND_NONE;
    vicuna_external_work_kind kind = VICUNA_EXTERNAL_WORK_BASH;
    llama_bash_tool_request bash_request = {};
    llama_cognitive_hard_memory_request hard_memory_request = {};
    llama_hard_memory_config hard_memory_config = {};
    llama_codex_tool_request codex_request = {};
    int64_t enqueued_ms = 0;
};

struct vicuna_external_work_result {
    int32_t command_id = -1;
    int32_t origin = -1;
    int32_t tool_kind = LLAMA_TOOL_KIND_NONE;
    vicuna_external_work_kind kind = VICUNA_EXTERNAL_WORK_BASH;
    llama_bash_tool_result bash_result = {};
    llama_cognitive_hard_memory_result hard_memory_result = {};
    llama_codex_tool_request codex_request = {};
    llama_codex_tool_result codex_result = {};
    int64_t completed_ms = 0;
};

struct vicuna_external_work_queue {
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<vicuna_external_work_item> pending;
    std::deque<vicuna_external_work_result> completed;
    std::unordered_set<int32_t> inflight_ids;
    bool stop = false;
};

struct vicuna_runtime_persistence_state {
    bool enabled = false;
    bool healthy = true;
    bool restore_attempted = false;
    bool restore_success = false;
    uint64_t persist_success_total = 0;
    uint64_t persist_fail_total = 0;
    int64_t last_persist_ms = 0;
    int64_t last_restore_ms = 0;
    std::string snapshot_path;
    std::string last_error;
};

struct vicuna_progress_snapshot {
    bool valid = false;
    int32_t active_extensions = 0;
    int32_t discovered_extensions = 0;
    int32_t permanent_extensions = 0;
    int32_t allostatic_extensions = 0;
    float allostatic_divergence = 0.0f;
    float promotion_readiness = 0.0f;
    float belief_pressure = 0.0f;
    uint64_t functional_update_total = 0;
    uint64_t process_update_total = 0;
};

struct vicuna_provenance_repository_state {
    bool enabled = false;
    bool healthy = true;
    uint64_t next_sequence = 1;
    uint64_t append_total = 0;
    uint64_t append_fail_total = 0;
    uint64_t active_loop_total = 0;
    uint64_t tool_result_total = 0;
    uint64_t dmn_total = 0;
    uint64_t discovered_increase_total = 0;
    uint64_t permanent_increase_total = 0;
    uint64_t allostatic_increase_total = 0;
    uint64_t functional_update_observed_total = 0;
    uint64_t process_update_observed_total = 0;
    uint64_t prune_total = 0;
    int64_t last_append_ms = 0;
    std::string path;
    std::string active_path;
    std::string session_id;
    std::string last_error;
    int64_t retention_ms = 48LL * 60LL * 60LL * 1000LL;
    int64_t active_bucket_start_ms = 0;
    bool has_last_snapshot = false;
    vicuna_progress_snapshot last_snapshot = {};
};

struct vicuna_external_observability {
    uint64_t bash_dispatch_total = 0;
    uint64_t bash_complete_total = 0;
    uint64_t bash_fail_total = 0;
    uint64_t hard_memory_dispatch_total = 0;
    uint64_t hard_memory_complete_total = 0;
    uint64_t hard_memory_fail_total = 0;
    uint64_t codex_dispatch_total = 0;
    uint64_t codex_complete_total = 0;
    uint64_t codex_fail_total = 0;
};

struct vicuna_mailbox_event {
    uint64_t sequence_number = 0;
    std::string response_id;
    json event;
};

struct vicuna_stored_response {
    std::string response_id;
    json response = json::object();
    std::vector<vicuna_mailbox_event> events;
    int64_t created_ms = 0;
    int64_t completed_ms = 0;
};

struct vicuna_proactive_mailbox {
    mutable std::mutex mutex;
    std::condition_variable cv;
    size_t max_responses = 64;
    uint64_t next_sequence_number = 1;
    std::deque<std::string> response_order;
    std::unordered_map<std::string, vicuna_stored_response> responses;
    std::deque<vicuna_mailbox_event> live_events;
    bool live_stream_connected = false;
    uint64_t publish_total = 0;
    uint64_t complete_total = 0;
    uint64_t fail_total = 0;
    uint64_t dropped_total = 0;
    int64_t last_publish_ms = 0;
};

struct vicuna_mailbox_snapshot {
    size_t max_responses = 64;
    uint64_t next_sequence_number = 1;
    std::deque<std::string> response_order;
    std::unordered_map<std::string, vicuna_stored_response> responses;
    std::deque<vicuna_mailbox_event> live_events;
    bool live_stream_connected = false;
    uint64_t publish_total = 0;
    uint64_t complete_total = 0;
    uint64_t fail_total = 0;
    uint64_t dropped_total = 0;
    int64_t last_publish_ms = 0;
};

struct vicuna_telegram_dialogue_turn {
    uint64_t turn_id = 0;
    std::string chat_scope;
    std::string user_text;
    std::string assistant_text;
    std::string assistant_source;
    std::string response_id;
    std::string dedupe_key;
    int64_t telegram_message_id = 0;
    int64_t updated_at_ms = 0;
};

struct vicuna_telegram_dialogue_history {
    mutable std::mutex mutex;
    size_t max_turns_per_scope = VICUNA_TELEGRAM_DIALOGUE_MAX_TURNS_PER_SCOPE;
    uint64_t next_turn_id = 1;
    std::unordered_map<std::string, std::deque<vicuna_telegram_dialogue_turn>> turns_by_scope;
    std::unordered_map<std::string, int64_t> latest_message_id_by_scope;
};

struct vicuna_telegram_dialogue_snapshot {
    size_t max_turns_per_scope = VICUNA_TELEGRAM_DIALOGUE_MAX_TURNS_PER_SCOPE;
    uint64_t next_turn_id = 1;
    std::unordered_map<std::string, std::deque<vicuna_telegram_dialogue_turn>> turns_by_scope;
    std::unordered_map<std::string, int64_t> latest_message_id_by_scope;
};

struct vicuna_telegram_outbox_item {
    uint64_t sequence_number = 0;
    std::string kind;
    std::string approval_id;
    std::string chat_scope;
    std::string dedupe_key;
    std::string text;
    int64_t reply_to_message_id = 0;
    std::string question;
    std::vector<std::string> options;
    int64_t created_at_ms = 0;
};

struct vicuna_telegram_outbox {
    mutable std::mutex mutex;
    size_t max_items = 64;
    uint64_t next_sequence_number = 1;
    std::deque<vicuna_telegram_outbox_item> items;
    uint64_t publish_total = 0;
    uint64_t dropped_total = 0;
    int64_t last_publish_ms = 0;
};

struct vicuna_telegram_outbox_snapshot {
    size_t max_items = 64;
    uint64_t next_sequence_number = 1;
    std::deque<vicuna_telegram_outbox_item> items;
    uint64_t publish_total = 0;
    uint64_t dropped_total = 0;
    int64_t last_publish_ms = 0;
};

enum vicuna_pending_approval_status {
    VICUNA_PENDING_APPROVAL_PENDING_PUBLISH = 0,
    VICUNA_PENDING_APPROVAL_AWAITING_USER = 1,
    VICUNA_PENDING_APPROVAL_APPROVED = 2,
    VICUNA_PENDING_APPROVAL_DENIED = 3,
    VICUNA_PENDING_APPROVAL_CANCELLED = 4,
    VICUNA_PENDING_APPROVAL_SUPERSEDED = 5,
    VICUNA_PENDING_APPROVAL_EXPIRED = 6,
    VICUNA_PENDING_APPROVAL_DISPATCH_FAILED = 7,
};

struct vicuna_pending_approval {
    std::string approval_id;
    int32_t command_id = -1;
    int32_t origin = -1;
    int32_t tool_kind = LLAMA_TOOL_KIND_NONE;
    int32_t tool_spec_index = -1;
    int32_t tool_job_id = -1;
    std::string capability_id;
    std::string tool_family_id;
    std::string method_name;
    std::string chat_scope;
    int64_t reply_to_message_id = 0;
    std::string question;
    std::vector<std::string> options;
    std::string dedupe_key;
    int32_t status = VICUNA_PENDING_APPROVAL_PENDING_PUBLISH;
    uint64_t published_sequence_number = 0;
    int64_t created_at_ms = 0;
    int64_t updated_at_ms = 0;
    int64_t resolved_at_ms = 0;
    std::string resolution_detail;
};

struct vicuna_foreground_consent_scope {
    std::string scope_id;
    int32_t task_id = -1;
    int32_t episode_id = 0;
    std::string chat_scope;
    int64_t telegram_message_id = 0;
    int64_t created_at_ms = 0;
    int64_t updated_at_ms = 0;
};

static std::string bounded_cstr_to_string(const char * value) {
    return value ? std::string(value) : std::string();
}

static std::string trim_ascii_copy(const std::string & value);

static const char * pending_approval_status_name(int32_t status) {
    switch (status) {
        case VICUNA_PENDING_APPROVAL_PENDING_PUBLISH:
            return "pending_publish";
        case VICUNA_PENDING_APPROVAL_AWAITING_USER:
            return "awaiting_user";
        case VICUNA_PENDING_APPROVAL_APPROVED:
            return "approved";
        case VICUNA_PENDING_APPROVAL_DENIED:
            return "denied";
        case VICUNA_PENDING_APPROVAL_CANCELLED:
            return "cancelled";
        case VICUNA_PENDING_APPROVAL_SUPERSEDED:
            return "superseded";
        case VICUNA_PENDING_APPROVAL_EXPIRED:
            return "expired";
        case VICUNA_PENDING_APPROVAL_DISPATCH_FAILED:
            return "dispatch_failed";
        default:
            return "unknown";
    }
}

static json pending_approval_to_json(const vicuna_pending_approval & approval) {
    return {
        {"approval_id", approval.approval_id},
        {"command_id", approval.command_id},
        {"origin", approval.origin},
        {"tool_kind", approval.tool_kind},
        {"tool_spec_index", approval.tool_spec_index},
        {"tool_job_id", approval.tool_job_id},
        {"capability_id", approval.capability_id},
        {"tool_family_id", approval.tool_family_id},
        {"method_name", approval.method_name},
        {"chat_scope", approval.chat_scope},
        {"reply_to_message_id", approval.reply_to_message_id},
        {"question", approval.question},
        {"options", approval.options},
        {"dedupe_key", approval.dedupe_key},
        {"status", pending_approval_status_name(approval.status)},
        {"published_sequence_number", approval.published_sequence_number},
        {"created_at_ms", approval.created_at_ms},
        {"updated_at_ms", approval.updated_at_ms},
        {"resolved_at_ms", approval.resolved_at_ms},
        {"resolution_detail", approval.resolution_detail},
    };
}

static bool pending_approval_from_json(const json & data, vicuna_pending_approval * out_approval) {
    if (!out_approval || !data.is_object()) {
        return false;
    }

    vicuna_pending_approval approval = {};
    approval.approval_id = trim_ascii_copy(json_value(data, "approval_id", std::string()));
    approval.command_id = json_value(data, "command_id", approval.command_id);
    approval.origin = json_value(data, "origin", approval.origin);
    approval.tool_kind = json_value(data, "tool_kind", approval.tool_kind);
    approval.tool_spec_index = json_value(data, "tool_spec_index", approval.tool_spec_index);
    approval.tool_job_id = json_value(data, "tool_job_id", approval.tool_job_id);
    approval.capability_id = trim_ascii_copy(json_value(data, "capability_id", std::string()));
    approval.tool_family_id = trim_ascii_copy(json_value(data, "tool_family_id", std::string()));
    approval.method_name = trim_ascii_copy(json_value(data, "method_name", std::string()));
    approval.chat_scope = trim_ascii_copy(json_value(data, "chat_scope", std::string()));
    approval.reply_to_message_id = std::max<int64_t>(0, json_value(data, "reply_to_message_id", int64_t(0)));
    approval.question = trim_ascii_copy(json_value(data, "question", std::string()));
    approval.dedupe_key = trim_ascii_copy(json_value(data, "dedupe_key", std::string()));
    approval.published_sequence_number =
            std::max<uint64_t>(0, json_value(data, "published_sequence_number", uint64_t(0)));
    approval.created_at_ms = std::max<int64_t>(0, json_value(data, "created_at_ms", int64_t(0)));
    approval.updated_at_ms = std::max<int64_t>(0, json_value(data, "updated_at_ms", int64_t(0)));
    approval.resolved_at_ms = std::max<int64_t>(0, json_value(data, "resolved_at_ms", int64_t(0)));
    approval.resolution_detail = trim_ascii_copy(json_value(data, "resolution_detail", std::string()));

    const std::string status = trim_ascii_copy(json_value(data, "status", std::string()));
    if (status == "pending_publish") {
        approval.status = VICUNA_PENDING_APPROVAL_PENDING_PUBLISH;
    } else if (status == "awaiting_user") {
        approval.status = VICUNA_PENDING_APPROVAL_AWAITING_USER;
    } else if (status == "approved") {
        approval.status = VICUNA_PENDING_APPROVAL_APPROVED;
    } else if (status == "denied") {
        approval.status = VICUNA_PENDING_APPROVAL_DENIED;
    } else if (status == "cancelled") {
        approval.status = VICUNA_PENDING_APPROVAL_CANCELLED;
    } else if (status == "superseded") {
        approval.status = VICUNA_PENDING_APPROVAL_SUPERSEDED;
    } else if (status == "expired") {
        approval.status = VICUNA_PENDING_APPROVAL_EXPIRED;
    } else if (status == "dispatch_failed") {
        approval.status = VICUNA_PENDING_APPROVAL_DISPATCH_FAILED;
    }

    if (data.contains("options") && data.at("options").is_array()) {
        for (const auto & option : data.at("options")) {
            const std::string value = trim_ascii_copy(option.is_string() ? option.get<std::string>() : std::string());
            if (!value.empty()) {
                approval.options.push_back(value);
            }
        }
    }

    if (approval.approval_id.empty() ||
            approval.command_id <= 0 ||
            approval.chat_scope.empty() ||
            approval.question.empty() ||
            approval.options.size() < 2) {
        return false;
    }

    *out_approval = std::move(approval);
    return true;
}

static json functional_activation_to_json(const llama_functional_activation_decision & activation) {
    json families = json::array();
    const int32_t family_count = std::min<int32_t>(activation.family_count, LLAMA_FUNCTIONAL_LORA_COUNT);
    for (int32_t family = 0; family < family_count; ++family) {
        families.push_back({
            {"family", family},
            {"gain", activation.gains[family]},
            {"predicted_gain", activation.predicted_gains[family]},
            {"sampled_noise", activation.sampled_noise[family]},
            {"bootstrap_std", activation.bootstrap_std[family]},
            {"bootstrap_perturbation", activation.bootstrap_perturbation[family]},
            {"hold_unit", activation.hold_unit[family]},
            {"hold_value", activation.hold_value[family]},
            {"priority", activation.priority[family]},
            {"reason_mask", activation.reason_mask[family]},
        });
    }
    return {
        {"loop_origin", activation.loop_origin},
        {"microphase", activation.microphase},
        {"top_family", activation.top_family},
        {"family_count", activation.family_count},
        {"activated_mask", activation.activated_mask},
        {"eligible_mask", activation.eligible_mask},
        {"exploration_std", activation.exploration_std},
        {"allostatic_distance", activation.allostatic_distance},
        {"allostatic_gradient_norm", activation.allostatic_gradient_norm},
        {"gating_invocation_count", activation.gating_invocation_count},
        {"families", std::move(families)},
    };
}

static json self_model_summary_to_json(const llama_self_model_state_info & model_state) {
    return {
        {"extension_summary", {
            {"active_count", model_state.extension_summary.active_count},
            {"transient_count", model_state.extension_summary.transient_count},
            {"permanent_count", model_state.extension_summary.permanent_count},
            {"discovered_count", model_state.extension_summary.discovered_count},
            {"gain_count", model_state.extension_summary.gain_count},
            {"allostatic_count", model_state.extension_summary.allostatic_count},
            {"hard_memory_count", model_state.extension_summary.hard_memory_count},
            {"tool_count", model_state.extension_summary.tool_count},
            {"mean_admission", model_state.extension_summary.mean_admission},
            {"mean_permanence", model_state.extension_summary.mean_permanence},
            {"mean_allostatic_eligibility", model_state.extension_summary.mean_allostatic_eligibility},
            {"context_activation", model_state.extension_summary.context_activation},
            {"gain_signal", model_state.extension_summary.gain_signal},
            {"gain_signal_abs", model_state.extension_summary.gain_signal_abs},
            {"allostatic_divergence", model_state.extension_summary.allostatic_divergence},
        }},
        {"belief_summary", {
            {"residual_allostatic_pressure", model_state.belief_summary.residual_allostatic_pressure},
            {"promotion_readiness", model_state.belief_summary.promotion_readiness},
            {"belief_entropy", model_state.belief_summary.belief_entropy},
            {"belief_confidence", model_state.belief_summary.belief_confidence},
            {"slot_pressure_mean", model_state.belief_summary.slot_pressure_mean},
            {"max_slot_pressure", model_state.belief_summary.max_slot_pressure},
        }},
        {"last_extension_trace", {
            {"valid", model_state.last_extension_trace.valid},
            {"candidate_count", model_state.last_extension_trace.candidate_count},
            {"promoted_count", model_state.last_extension_trace.promoted_count},
            {"winner_index", model_state.last_extension_trace.winner_index},
        }},
        {"forecast", {
            {"valid", model_state.forecast.valid},
            {"predicted_steps_remaining", model_state.forecast.predicted_steps_remaining},
            {"predicted_inference_cost_remaining", model_state.forecast.predicted_inference_cost_remaining},
            {"predicted_satisfaction_delta", model_state.forecast.predicted_satisfaction_delta},
            {"predicted_recovery_delta", model_state.forecast.predicted_recovery_delta},
            {"predicted_goal_progress_delta", model_state.forecast.predicted_goal_progress_delta},
            {"confidence", model_state.forecast.confidence},
        }},
        {"prediction_error", {
            {"valid", model_state.prediction_error.valid},
            {"steps_error", model_state.prediction_error.steps_error},
            {"inference_cost_error", model_state.prediction_error.inference_cost_error},
            {"satisfaction_error", model_state.prediction_error.satisfaction_error},
            {"recovery_error", model_state.prediction_error.recovery_error},
            {"goal_progress_error", model_state.prediction_error.goal_progress_error},
        }},
    };
}

static json functional_trace_summary_to_json(const llama_functional_lora_trace & trace) {
    json families = json::array();
    for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
        const auto & state = trace.family_state[family];
        families.push_back({
            {"family", state.family},
            {"enabled", state.enabled},
            {"compatible", state.compatible},
            {"active_now", state.active_now},
            {"current_gain", state.current_gain},
            {"predicted_gain", state.predicted_gain},
            {"current_microphase", state.current_microphase},
            {"activation_count", state.activation_count},
            {"update_count", state.update_count},
            {"last_signed_outcome", state.last_signed_outcome},
            {"last_meta_loss", state.last_meta_loss},
        });
    }
    return {
        {"activation", functional_activation_to_json(trace.last_activation)},
        {"families", std::move(families)},
    };
}

static json process_trace_summary_to_json(const llama_process_functional_trace & trace) {
    return {
        {"valid", trace.valid},
        {"matched_existing_entry", trace.matched_existing_entry},
        {"matched_entry_slot", trace.matched_entry_slot},
        {"created_entry", trace.created_entry},
        {"created_entry_slot", trace.created_entry_slot},
        {"creation_reason", trace.creation_reason},
        {"evicted_entry_slot", trace.evicted_entry_slot},
        {"signed_outcome", trace.signed_outcome},
        {"magnitude", trace.magnitude},
        {"weak_or_worse_ratio", trace.weak_or_worse_ratio},
        {"bank_size", trace.bank_size},
        {"bank_capacity", trace.bank_capacity},
        {"activation_attached", trace.activation_attached},
        {"signature", {
            {"valid", trace.signature.valid},
            {"signature_hash", trace.signature.signature_hash},
            {"scope_kind", trace.signature.scope_kind},
            {"family", trace.signature.family},
            {"loop_origin", trace.signature.loop_origin},
            {"microphase", trace.signature.microphase},
            {"plan_mode", trace.signature.plan_mode},
            {"plan_step_kind", trace.signature.plan_step_kind},
            {"tool_kind", trace.signature.tool_kind},
            {"source_family", trace.signature.source_family},
            {"requires_tool_result", trace.signature.requires_tool_result},
            {"tool_name", bounded_cstr_to_string(trace.signature.tool_name)},
            {"capability_id", bounded_cstr_to_string(trace.signature.capability_id)},
            {"provenance_namespace", bounded_cstr_to_string(trace.signature.provenance_namespace)},
            {"semantic_key", bounded_cstr_to_string(trace.signature.semantic_key)},
        }},
    };
}

static json cognitive_loop_state_to_json(const llama_cognitive_loop_state & state) {
    return {
        {"phase", state.phase},
        {"terminal_reason", state.terminal_reason},
        {"max_steps", state.max_steps},
        {"steps_taken", state.steps_taken},
        {"continuation_allowed", state.continuation_allowed},
        {"waiting_on_tool", state.waiting_on_tool},
        {"tool_registry_count", state.tool_registry_count},
    };
}

static json cognitive_plan_step_to_json(const llama_cognitive_plan_step & step, int32_t index) {
    return {
        {"index", index},
        {"kind", step.kind},
        {"status", step.status},
        {"tool_kind", step.tool_kind},
        {"tool_spec_index", step.tool_spec_index},
        {"source_family", step.source_family},
        {"reason_mask", step.reason_mask},
        {"priority", step.priority},
        {"expected_steps", step.expected_steps},
        {"requires_tool_result", step.requires_tool_result},
        {"capability_id", bounded_cstr_to_string(step.capability_id)},
        {"provenance_namespace", bounded_cstr_to_string(step.provenance_namespace)},
    };
}

static json cognitive_plan_trace_to_json(const llama_cognitive_plan_trace & trace) {
    json steps = json::array();
    for (int32_t i = 0; i < trace.step_count && i < LLAMA_COGNITIVE_MAX_PLAN_STEPS; ++i) {
        steps.push_back(cognitive_plan_step_to_json(trace.steps[i], i));
    }
    return {
        {"valid", trace.valid},
        {"plan_id", trace.plan_id},
        {"origin", trace.origin},
        {"mode", trace.mode},
        {"status", trace.status},
        {"revision_count", trace.revision_count},
        {"current_step_index", trace.current_step_index},
        {"step_count", trace.step_count},
        {"selected_family", trace.selected_family},
        {"terminal_reason", trace.terminal_reason},
        {"reason_mask", trace.reason_mask},
        {"plan_score", trace.plan_score},
        {"ambiguity", trace.ambiguity},
        {"steps", std::move(steps)},
    };
}

static json active_candidates_to_json(const llama_active_loop_trace & trace) {
    json candidates = json::array();
    for (int32_t i = 0; i < trace.candidate_count && i < LLAMA_ACTIVE_LOOP_MAX_CANDIDATES; ++i) {
        const auto & candidate = trace.candidates[i];
        candidates.push_back({
            {"action", candidate.action},
            {"score", candidate.score},
            {"user_relevance", candidate.user_relevance},
            {"latency_pressure", candidate.latency_pressure},
            {"tool_affinity", candidate.tool_affinity},
            {"inhibition", candidate.inhibition},
            {"reason_mask", candidate.reason_mask},
        });
    }
    return candidates;
}

static json dmn_candidates_to_json(const llama_dmn_tick_trace & trace) {
    json candidates = json::array();
    for (int32_t i = 0; i < trace.candidate_count && i < LLAMA_DMN_MAX_CANDIDATES; ++i) {
        const auto & candidate = trace.candidates[i];
        candidates.push_back({
            {"action", candidate.action},
            {"score", candidate.score},
            {"inhibition", candidate.inhibition},
            {"social_relevance", candidate.social_relevance},
            {"continuation", candidate.continuation},
            {"tool_affinity", candidate.tool_affinity},
            {"reason_mask", candidate.reason_mask},
        });
    }
    return candidates;
}

static json dmn_reactivation_targets_to_json(const llama_dmn_tick_trace & trace) {
    json targets = json::array();
    for (int32_t i = 0; i < trace.reactivation_count && i < LLAMA_DMN_MAX_REACTIVATION_TARGETS; ++i) {
        const auto & target = trace.reactivation_targets[i];
        targets.push_back({
            {"handle_id", target.handle_id},
            {"kind", target.kind},
            {"priority", target.priority},
            {"top_similarity", target.top_similarity},
            {"last_update_monotonic_ms", target.last_update_monotonic_ms},
        });
    }
    return targets;
}

static json active_trace_summary_to_json(const llama_active_loop_trace & trace) {
    return {
        {"episode_id", trace.episode_id},
        {"source_role", trace.source_role},
        {"channel", trace.channel},
        {"event_flags", trace.event_flags},
        {"arrival_time_us", trace.arrival_time_us},
        {"completed_time_us", trace.completed_time_us},
        {"shared_state_version", trace.shared_state_version},
        {"deferred_background", trace.deferred_background},
        {"emit_allowed", trace.emit_allowed},
        {"emit_noted", trace.emit_noted},
        {"tool_followup_expected", trace.tool_followup_expected},
        {"winner_action", trace.winner_action},
        {"winner_score", trace.winner_score},
        {"runner_up_action", trace.runner_up_action},
        {"runner_up_score", trace.runner_up_score},
        {"reason_mask", trace.reason_mask},
        {"loop_state", cognitive_loop_state_to_json(trace.loop_state)},
        {"candidate_count", trace.candidate_count},
        {"candidates", active_candidates_to_json(trace)},
        {"plan", cognitive_plan_trace_to_json(trace.plan)},
        {"functional_activation", functional_activation_to_json(trace.functional_activation)},
        {"tool_proposal", {
            {"valid", trace.tool_proposal.valid},
            {"tool_kind", trace.tool_proposal.tool_kind},
            {"spec_index", trace.tool_proposal.spec_index},
            {"reason_mask", trace.tool_proposal.reason_mask},
            {"source_family", trace.tool_proposal.source_family},
            {"expected_steps", trace.tool_proposal.expected_steps},
            {"expected_observation_gain", trace.tool_proposal.expected_observation_gain},
            {"job_id", trace.tool_proposal.job_id},
            {"capability_id", bounded_cstr_to_string(trace.tool_proposal.capability_id)},
            {"provenance_namespace", bounded_cstr_to_string(trace.tool_proposal.provenance_namespace)},
        }},
        {"observation", {
            {"valid", trace.observation.valid},
            {"tool_kind", trace.observation.tool_kind},
            {"spec_index", trace.observation.spec_index},
            {"job_id", trace.observation.job_id},
            {"status", trace.observation.status},
            {"signal", trace.observation.signal},
            {"followup_affinity", trace.observation.followup_affinity},
            {"capability_id", bounded_cstr_to_string(trace.observation.capability_id)},
            {"provenance_namespace", bounded_cstr_to_string(trace.observation.provenance_namespace)},
        }},
    };
}

static json dmn_trace_summary_to_json(const llama_dmn_tick_trace & trace) {
    return {
        {"tick_id", trace.tick_id},
        {"admitted", trace.admitted},
        {"deferred_for_foreground", trace.deferred_for_foreground},
        {"pressure", {
            {"contradiction", trace.pressure.contradiction},
            {"uncertainty", trace.pressure.uncertainty},
            {"reactivation", trace.pressure.reactivation},
            {"goals", trace.pressure.goals},
            {"tool_delta", trace.pressure.tool_delta},
            {"counterfactual", trace.pressure.counterfactual},
            {"continuation", trace.pressure.continuation},
            {"repair", trace.pressure.repair},
            {"total", trace.pressure.total},
        }},
        {"candidate_count", trace.candidate_count},
        {"candidates", dmn_candidates_to_json(trace)},
        {"winner_action", trace.winner_action},
        {"winner_score", trace.winner_score},
        {"runner_up_action", trace.runner_up_action},
        {"runner_up_score", trace.runner_up_score},
        {"burst_count", trace.burst_count},
        {"maintenance_mask", trace.maintenance_mask},
        {"tool_kind", trace.tool_kind},
        {"tool_spec_index", trace.tool_spec_index},
        {"tool_job_id", trace.tool_job_id},
        {"loop_state", cognitive_loop_state_to_json(trace.loop_state)},
        {"reactivation_count", trace.reactivation_count},
        {"reactivation_targets", dmn_reactivation_targets_to_json(trace)},
        {"seed_source_mask", trace.seed_source_mask},
        {"self_model_revision", {
            {"valid", trace.self_model_revision.valid},
            {"revision_id", trace.self_model_revision.revision_id},
            {"input_hash", trace.self_model_revision.input_hash},
            {"materiality_score", trace.self_model_revision.materiality_score},
            {"requires_prompt_regen", trace.self_model_revision.requires_prompt_regen},
        }},
        {"favorable_divergence", trace.favorable_divergence},
        {"plan", cognitive_plan_trace_to_json(trace.plan)},
        {"functional_activation", functional_activation_to_json(trace.functional_activation)},
        {"tool_proposal", {
            {"valid", trace.tool_proposal.valid},
            {"tool_kind", trace.tool_proposal.tool_kind},
            {"spec_index", trace.tool_proposal.spec_index},
            {"reason_mask", trace.tool_proposal.reason_mask},
            {"source_family", trace.tool_proposal.source_family},
            {"expected_steps", trace.tool_proposal.expected_steps},
            {"expected_observation_gain", trace.tool_proposal.expected_observation_gain},
            {"job_id", trace.tool_proposal.job_id},
            {"capability_id", bounded_cstr_to_string(trace.tool_proposal.capability_id)},
            {"provenance_namespace", bounded_cstr_to_string(trace.tool_proposal.provenance_namespace)},
        }},
        {"observation", {
            {"valid", trace.observation.valid},
            {"tool_kind", trace.observation.tool_kind},
            {"spec_index", trace.observation.spec_index},
            {"job_id", trace.observation.job_id},
            {"status", trace.observation.status},
            {"signal", trace.observation.signal},
            {"followup_affinity", trace.observation.followup_affinity},
            {"capability_id", bounded_cstr_to_string(trace.observation.capability_id)},
            {"provenance_namespace", bounded_cstr_to_string(trace.observation.provenance_namespace)},
        }},
    };
}

static json counterfactual_trace_summary_to_json(const llama_counterfactual_trace & trace) {
    json candidates = json::array();
    for (int32_t i = 0; i < trace.candidate_count; ++i) {
        const auto & candidate = trace.candidates[i];
        candidates.push_back({
            {"family", candidate.family},
            {"risk_tier", candidate.risk_tier},
            {"subject_id", candidate.subject_id},
            {"functional_target_kind", candidate.functional_target_kind},
            {"functional_family", candidate.functional_family},
            {"process_entry_slot", candidate.process_entry_slot},
            {"proposal_family", candidate.proposal_family},
            {"replay_mode", candidate.replay_mode},
            {"snapshot_slot", candidate.snapshot_slot},
            {"expected_improvement", candidate.expected_improvement},
            {"confidence", candidate.confidence},
            {"fragility_penalty", candidate.fragility_penalty},
            {"concentration_penalty", candidate.concentration_penalty},
            {"robustness_score", candidate.robustness_score},
            {"orthogonality", candidate.orthogonality},
            {"realized_score", candidate.realized_score},
            {"signed_advantage_vs_current", candidate.signed_advantage_vs_current},
        });
    }
    return {
        {"candidate_count", trace.candidate_count},
        {"winner_index", trace.winner_index},
        {"escalated", trace.escalated},
        {"escalation_family", trace.escalation_family},
        {"candidates", std::move(candidates)},
    };
}

static json governance_trace_summary_to_json(const llama_governance_trace & trace) {
    return {
        {"proposal_family", trace.proposal_family},
        {"risk_tier", trace.risk_tier},
        {"outcome", trace.outcome},
        {"evidence", trace.evidence},
        {"threshold", trace.threshold},
        {"dissatisfaction", trace.dissatisfaction},
        {"recent_user_valence", trace.recent_user_valence},
        {"repair_rendered", trace.repair_rendered},
        {"repair_message_length", trace.repair_message_length},
        {"repair_message", bounded_cstr_to_string(trace.repair_message)},
    };
}

static json remediation_plan_to_json(const llama_remediation_plan & plan) {
    return {
        {"action", plan.action},
        {"source_family", plan.source_family},
        {"tool_kind", plan.tool_kind},
        {"expected_improvement", plan.expected_improvement},
        {"confidence", plan.confidence},
        {"budget", plan.budget},
        {"tool_job_id", plan.tool_job_id},
        {"applied", plan.applied},
        {"pre_divergence", plan.pre_divergence},
        {"post_divergence", plan.post_divergence},
    };
}

static json bash_request_to_json(const llama_bash_tool_request & request) {
    return {
        {"command_id", request.command_id},
        {"origin", request.origin},
        {"tool_job_id", request.tool_job_id},
        {"timeout_ms", request.timeout_ms},
        {"cpu_time_limit_secs", request.cpu_time_limit_secs},
        {"max_child_processes", request.max_child_processes},
        {"max_open_files", request.max_open_files},
        {"max_file_size_bytes", request.max_file_size_bytes},
        {"max_stdout_bytes", request.max_stdout_bytes},
        {"max_stderr_bytes", request.max_stderr_bytes},
        {"inherit_env", request.inherit_env},
        {"login_shell", request.login_shell},
        {"reject_shell_metacharacters", request.reject_shell_metacharacters},
        {"command_ready", request.command_ready},
        {"bash_path", bounded_cstr_to_string(request.bash_path)},
        {"working_directory", bounded_cstr_to_string(request.working_directory)},
        {"allowed_commands", bounded_cstr_to_string(request.allowed_commands)},
        {"blocked_patterns", bounded_cstr_to_string(request.blocked_patterns)},
        {"allowed_env", bounded_cstr_to_string(request.allowed_env)},
        {"intent_text", bounded_cstr_to_string(request.intent_text)},
        {"command_text", bounded_cstr_to_string(request.command_text)},
    };
}

static json hard_memory_request_to_json(const llama_cognitive_hard_memory_request & request) {
    json out = {
        {"command_id", request.command_id},
        {"origin", request.origin},
        {"tool_job_id", request.tool_job_id},
        {"operation", request.operation},
        {"container_tag", bounded_cstr_to_string(request.container_tag)},
        {"write_count", request.write_count},
        {"query", {
            {"limit", request.query.limit},
            {"threshold", request.query.threshold},
            {"include_profile", request.query.include_profile},
            {"use_temporal_self_hint", request.query.use_temporal_self_hint},
            {"temporal_adapter_role", request.query.temporal_adapter_role},
            {"query", bounded_cstr_to_string(request.query.query)},
            {"container_tag", bounded_cstr_to_string(request.query.container_tag)},
        }},
    };
    json write_items = json::array();
    for (int32_t i = 0; i < request.write_count && i < LLAMA_HARD_MEMORY_MAX_PRIMITIVES; ++i) {
        const auto & item = request.write_items[i];
        json tags = json::array();
        for (int32_t j = 0; j < LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS; ++j) {
            const std::string tag = bounded_cstr_to_string(item.primitive.tags[j]);
            if (!tag.empty()) {
                tags.push_back(tag);
            }
        }
        write_items.push_back({
            {"is_static", item.is_static},
            {"primitive", {
                {"kind", item.primitive.kind},
                {"domain", item.primitive.domain},
                {"source_role", item.primitive.source_role},
                {"source_channel", item.primitive.source_channel},
                {"source_tool_kind", item.primitive.source_tool_kind},
                {"transaction_id", item.primitive.transaction_id},
                {"flags", item.primitive.flags},
                {"importance", item.primitive.importance},
                {"confidence", item.primitive.confidence},
                {"gain_bias", item.primitive.gain_bias},
                {"allostatic_relevance", item.primitive.allostatic_relevance},
                {"key", bounded_cstr_to_string(item.primitive.key)},
                {"title", bounded_cstr_to_string(item.primitive.title)},
                {"content", bounded_cstr_to_string(item.primitive.content)},
                {"tags", std::move(tags)},
            }},
        });
    }
    out["write_items"] = std::move(write_items);
    return out;
}

static json codex_request_to_json(const llama_codex_tool_request & request) {
    return {
        {"command_id", request.command_id},
        {"origin", request.origin},
        {"tool_job_id", request.tool_job_id},
        {"timeout_ms", request.timeout_ms},
        {"max_stdout_bytes", request.max_stdout_bytes},
        {"max_stderr_bytes", request.max_stderr_bytes},
        {"dangerous_no_approval", request.dangerous_no_approval},
        {"rebuild_after_changes", request.rebuild_after_changes},
        {"verify_tool_access_after_rebuild", request.verify_tool_access_after_rebuild},
        {"command_ready", request.command_ready},
        {"codex_path", bounded_cstr_to_string(request.codex_path)},
        {"working_directory", bounded_cstr_to_string(request.working_directory)},
        {"rebuild_script_path", bounded_cstr_to_string(request.rebuild_script_path)},
        {"rebuild_helper_path", bounded_cstr_to_string(request.rebuild_helper_path)},
        {"completion_message_path", bounded_cstr_to_string(request.completion_message_path)},
        {"intent_text", bounded_cstr_to_string(request.intent_text)},
        {"task_prompt", bounded_cstr_to_string(request.task_prompt)},
    };
}

static json telegram_relay_request_to_json(const llama_telegram_relay_request & request) {
    return {
        {"command_id", request.command_id},
        {"origin", request.origin},
        {"tool_job_id", request.tool_job_id},
        {"intent_kind", request.intent_kind},
        {"urgency", request.urgency},
        {"command_ready", request.command_ready},
        {"dedupe_key", bounded_cstr_to_string(request.dedupe_key)},
        {"text", bounded_cstr_to_string(request.text)},
    };
}

static json telegram_ask_options_request_to_json(const llama_telegram_ask_options_request & request) {
    json options = json::array();
    const int32_t option_count = std::clamp(request.option_count, 0, (int32_t) LLAMA_TELEGRAM_ASK_MAX_OPTIONS);
    for (int32_t i = 0; i < option_count; ++i) {
        options.push_back(bounded_cstr_to_string(request.options[i].label));
    }
    return {
        {"command_id", request.command_id},
        {"origin", request.origin},
        {"tool_job_id", request.tool_job_id},
        {"urgency", request.urgency},
        {"command_ready", request.command_ready},
        {"option_count", option_count},
        {"dedupe_key", bounded_cstr_to_string(request.dedupe_key)},
        {"chat_scope", bounded_cstr_to_string(request.chat_scope)},
        {"question", bounded_cstr_to_string(request.question)},
        {"options", std::move(options)},
    };
}

static json bash_result_to_json(const llama_bash_tool_result & result) {
    json out = {
        {"command_id", result.command_id},
        {"tool_job_id", result.tool_job_id},
        {"exit_code", result.exit_code},
        {"term_signal", result.term_signal},
        {"runtime_ms", result.runtime_ms},
        {"timed_out", result.timed_out},
        {"launch_failed", result.launch_failed},
        {"truncated_stdout", result.truncated_stdout},
        {"truncated_stderr", result.truncated_stderr},
    };
    const std::string stdout_text = bounded_cstr_to_string(result.stdout_text);
    const std::string stderr_text = bounded_cstr_to_string(result.stderr_text);
    const std::string error_text = bounded_cstr_to_string(result.error_text);
    if (!stdout_text.empty()) {
        out["stdout_text"] = stdout_text;
    }
    if (!stderr_text.empty()) {
        out["stderr_text"] = stderr_text;
    }
    if (!error_text.empty()) {
        out["error_text"] = error_text;
    }
    return out;
}

static json hard_memory_result_to_json(const llama_cognitive_hard_memory_result & result) {
    json out = {
        {"command_id", result.command_id},
        {"tool_job_id", result.tool_job_id},
        {"operation", result.operation},
    };
    if (result.operation == LLAMA_COG_HARD_MEMORY_OPERATION_WRITE) {
        out["archived"] = result.archive_trace.archived;
        out["attempted"] = result.archive_trace.attempted;
        out["status_code"] = result.archive_trace.status_code;
        out["primitive_count"] = result.archive_trace.primitive_count;
        out["request_started_us"] = result.archive_trace.request_started_us;
        out["request_completed_us"] = result.archive_trace.request_completed_us;
    } else {
        out["ok"] = result.result.ok;
        out["status_code"] = result.result.status_code;
        out["result_count"] = result.result.result_count;
        out["request_started_us"] = result.result.request_started_us;
        out["request_completed_us"] = result.result.request_completed_us;
        out["mean_similarity"] = result.result.retrieval_summary.mean_similarity;
        out["gain_support"] = result.result.retrieval_summary.gain_support;
        out["allostatic_support"] = result.result.retrieval_summary.allostatic_support;
    }
    return out;
}

static json codex_result_to_json(const llama_codex_tool_result & result) {
    return {
        {"command_id", result.command_id},
        {"tool_job_id", result.tool_job_id},
        {"exit_code", result.exit_code},
        {"runtime_ms", result.runtime_ms},
        {"launch_failed", result.launch_failed},
        {"repo_changed", result.repo_changed},
        {"rebuild_attempted", result.rebuild_attempted},
        {"rebuild_succeeded", result.rebuild_succeeded},
        {"accessibility_verified", result.accessibility_verified},
        {"truncated_stdout", result.truncated_stdout},
        {"truncated_stderr", result.truncated_stderr},
    };
}

static json telegram_relay_result_to_json(const llama_telegram_relay_result & result) {
    return {
        {"command_id", result.command_id},
        {"tool_job_id", result.tool_job_id},
        {"intent_kind", result.intent_kind},
        {"delivered", result.delivered},
        {"delivered_at_ms", result.delivered_at_ms},
        {"dedupe_key", bounded_cstr_to_string(result.dedupe_key)},
        {"error_text", bounded_cstr_to_string(result.error_text)},
    };
}

static json telegram_ask_options_result_to_json(const llama_telegram_ask_options_result & result) {
    return {
        {"command_id", result.command_id},
        {"tool_job_id", result.tool_job_id},
        {"delivered", result.delivered},
        {"delivered_at_ms", result.delivered_at_ms},
        {"dedupe_key", bounded_cstr_to_string(result.dedupe_key)},
        {"chat_scope", bounded_cstr_to_string(result.chat_scope)},
        {"error_text", bounded_cstr_to_string(result.error_text)},
    };
}

static std::string trim_ascii_copy(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace((unsigned char) value[begin])) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace((unsigned char) value[end - 1])) {
        --end;
    }
    return value.substr(begin, end - begin);
}

static std::string log_excerpt(const std::string & value, size_t max_chars = 1600) {
    return bounded_runtime_log_excerpt(value, max_chars);
}

static void split_hidden_reasoning_text(
        const llama_vocab * vocab,
        const std::string & text,
        std::string * out_reasoning,
        std::string * out_visible,
        std::vector<std::string> * out_reasoning_blocks = nullptr) {
    if (out_reasoning) {
        out_reasoning->clear();
    }
    if (out_visible) {
        out_visible->clear();
    }
    if (out_reasoning_blocks) {
        out_reasoning_blocks->clear();
    }

    static const std::string think_open = "<think>";
    static const std::string think_close = "</think>";

    std::string visible;
    std::string reasoning;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t think_start = text.find(think_open, pos);
        if (think_start == std::string::npos) {
            visible += text.substr(pos);
            break;
        }

        visible += text.substr(pos, think_start - pos);
        const size_t body_start = think_start + think_open.size();
        const size_t think_end = text.find(think_close, body_start);
        if (think_end == std::string::npos) {
            visible += text.substr(think_start);
            break;
        }

        const std::string chunk = trim_ascii_copy(text.substr(body_start, think_end - body_start));
        if (!chunk.empty()) {
            const std::string bounded_chunk = truncate_text_to_token_budget(
                    vocab,
                    chunk,
                    SERVER_THINK_BLOCK_TOKEN_CAP);
            if (!bounded_chunk.empty()) {
                if (out_reasoning_blocks) {
                    out_reasoning_blocks->push_back(bounded_chunk);
                }
                if (!reasoning.empty()) {
                    reasoning += "\n\n";
                }
                reasoning += bounded_chunk;
            }
        }
        pos = think_end + think_close.size();
    }

    visible = trim_ascii_copy(visible);
    reasoning = trim_ascii_copy(reasoning);
    if (reasoning.empty() && !visible.empty()) {
        reasoning = visible;
        visible.clear();
    }

    if (out_reasoning) {
        *out_reasoning = std::move(reasoning);
    }
    if (out_visible) {
        *out_visible = std::move(visible);
    }
}

static int32_t parse_hard_memory_kind_json(const json & value) {
    const std::string name = value.is_string() ? trim_ascii_copy(value.get<std::string>()) : std::string();
    if (name == "trajectory") return LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY;
    if (name == "outcome") return LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME;
    if (name == "tool_observation") return LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_OBSERVATION;
    if (name == "user_model") return LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL;
    if (name == "self_model_fragment") return LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT;
    if (name == "event_fragment") return LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT;
    return value.is_number_integer() ? value.get<int32_t>() : LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME;
}

static int32_t parse_hard_memory_domain_json(const json & value) {
    const std::string name = value.is_string() ? trim_ascii_copy(value.get<std::string>()) : std::string();
    if (name == "goal_progress") return LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS;
    if (name == "user_outcome") return LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME;
    if (name == "epistemic") return LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
    if (name == "efficiency") return LLAMA_HARD_MEMORY_DOMAIN_EFFICIENCY;
    if (name == "recovery") return LLAMA_HARD_MEMORY_DOMAIN_RECOVERY;
    if (name == "strategy") return LLAMA_HARD_MEMORY_DOMAIN_STRATEGY;
    if (name == "self_improvement") return LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT;
    return value.is_number_integer() ? value.get<int32_t>() : LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC;
}

static bool populate_hard_memory_write_item_from_json(
        const json & item_json,
        llama_hard_memory_write_item * out_item) {
    if (!out_item || !item_json.is_object()) {
        return false;
    }

    const std::string content = trim_ascii_copy(json_value(item_json, "content", std::string()));
    if (content.empty()) {
        return false;
    }

    *out_item = {};
    out_item->is_static = json_value(item_json, "isStatic", false);
    out_item->primitive = llama_hard_memory_default_primitive();
    out_item->primitive.kind = parse_hard_memory_kind_json(item_json.value("kind", json("outcome")));
    out_item->primitive.domain = parse_hard_memory_domain_json(item_json.value("domain", json("epistemic")));
    out_item->primitive.source_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
    out_item->primitive.source_channel = LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY;
    out_item->primitive.source_tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_WRITE;
    out_item->primitive.flags =
            LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_GAIN |
            LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_DERIVED;
    out_item->primitive.importance = json_value(item_json, "importance", out_item->primitive.importance);
    out_item->primitive.confidence = json_value(item_json, "confidence", out_item->primitive.confidence);
    out_item->primitive.gain_bias = json_value(item_json, "gainBias", out_item->primitive.gain_bias);
    out_item->primitive.allostatic_relevance = json_value(item_json, "allostaticRelevance", out_item->primitive.allostatic_relevance);
    std::snprintf(out_item->primitive.key, sizeof(out_item->primitive.key), "%s", json_value(item_json, "key", std::string()).c_str());
    std::snprintf(out_item->primitive.title, sizeof(out_item->primitive.title), "%s", json_value(item_json, "title", std::string()).c_str());
    std::snprintf(out_item->primitive.content, sizeof(out_item->primitive.content), "%s", content.c_str());

    const json tags = item_json.value("tags", json::array());
    if (tags.is_array()) {
        int32_t written = 0;
        for (const auto & tag : tags) {
            if (written >= LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS || !tag.is_string()) {
                break;
            }
            const std::string tag_text = trim_ascii_copy(tag.get<std::string>());
            if (tag_text.empty()) {
                continue;
            }
            std::snprintf(out_item->primitive.tags[written], sizeof(out_item->primitive.tags[written]), "%s", tag_text.c_str());
            ++written;
        }
    }

    return true;
}

static int32_t telegram_intent_kind_from_text(const std::string & value) {
    const std::string normalized = trim_ascii_copy(value);
    if (normalized == "question") {
        return LLAMA_TELEGRAM_RELAY_QUESTION;
    }
    if (normalized == "conclusion") {
        return LLAMA_TELEGRAM_RELAY_CONCLUSION;
    }
    return LLAMA_TELEGRAM_RELAY_COMMENT;
}

static std::string request_header_value_ci(const server_http_req & req, const std::string & key) {
    auto fold_ascii = [](const std::string & input) {
        std::string folded = trim_ascii_copy(input);
        std::transform(
                folded.begin(),
                folded.end(),
                folded.begin(),
                [](unsigned char ch) { return (char) std::tolower(ch); });
        return folded;
    };

    const std::string wanted = fold_ascii(key);
    for (const auto & [header_key, header_value] : req.headers) {
        if (fold_ascii(header_key) == wanted) {
            return trim_ascii_copy(header_value);
        }
    }
    return {};
}

static int64_t parse_int64_header_value(const std::string & value) {
    if (value.empty()) {
        return 0;
    }
    try {
        return std::stoll(value);
    } catch (...) {
        return 0;
    }
}

static bool parse_bool_header_value(const std::string & value) {
    if (value.empty()) {
        return false;
    }
    std::string normalized = trim_ascii_copy(value);
    std::transform(
            normalized.begin(),
            normalized.end(),
            normalized.begin(),
            [](unsigned char ch) { return (char) std::tolower(ch); });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

static std::string extract_chat_message_text_content(const json & content) {
    if (content.is_string()) {
        return trim_ascii_copy(content.get<std::string>());
    }
    if (!content.is_array()) {
        return {};
    }

    std::string joined;
    for (const auto & part : content) {
        if (!part.is_object()) {
            continue;
        }
        const std::string part_type = trim_ascii_copy(json_value(part, "type", std::string("text")));
        if (part_type != "text") {
            continue;
        }
        const std::string text = trim_ascii_copy(json_value(part, "text", std::string()));
        if (text.empty()) {
            continue;
        }
        if (!joined.empty()) {
            joined.push_back('\n');
        }
        joined += text;
    }
    return joined;
}

static std::vector<common_chat_msg> extract_telegram_transcript_messages(const std::string & raw_body) {
    std::vector<common_chat_msg> messages;
    if (raw_body.empty()) {
        return messages;
    }

    try {
        const json body = json::parse(raw_body);
        if (!body.contains("messages") || !body.at("messages").is_array()) {
            return messages;
        }
        for (const auto & entry : body.at("messages")) {
            if (!entry.is_object()) {
                continue;
            }
            const std::string role = trim_ascii_copy(json_value(entry, "role", std::string()));
            if (role != "user" && role != "assistant") {
                continue;
            }
            const std::string content = extract_chat_message_text_content(entry.value("content", json()));
            if (content.empty()) {
                continue;
            }
            common_chat_msg msg;
            msg.role = role;
            msg.content = content;
            messages.push_back(std::move(msg));
        }
    } catch (...) {
    }

    return messages;
}

static std::string percent_encode_query(const std::string & text) {
    static const char hex[] = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(text.size() * 3);
    for (unsigned char ch : text) {
        if ((ch >= 'A' && ch <= 'Z') ||
                (ch >= 'a' && ch <= 'z') ||
                (ch >= '0' && ch <= '9') ||
                ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            encoded.push_back((char) ch);
        } else {
            encoded.push_back('%');
            encoded.push_back(hex[(ch >> 4) & 0x0F]);
            encoded.push_back(hex[ch & 0x0F]);
        }
    }
    return encoded;
}

static std::string format_react_bash_observation(const llama_bash_tool_result & result) {
    const std::string stdout_text = trim_ascii_copy(result.stdout_text);
    const std::string stderr_text = trim_ascii_copy(result.stderr_text);
    const std::string error_text = trim_ascii_copy(result.error_text);

    auto append_truncation_note = [&](std::string body) {
        if (result.truncated_stdout || result.truncated_stderr) {
            if (!body.empty()) {
                body += "\n\n";
            }
            body += "[tool output truncated at capture limit]";
        }
        return body;
    };

    const bool success =
            !result.launch_failed &&
            !result.timed_out &&
            result.exit_code == 0;

    if (success) {
        if (!stdout_text.empty()) {
            return append_truncation_note(stdout_text);
        }
        if (!stderr_text.empty()) {
            return append_truncation_note(stderr_text);
        }
        if (!error_text.empty()) {
            return error_text;
        }
        return "success";
    }

    std::ostringstream oss;
    oss << "tool execution failed";
    if (result.launch_failed) {
        oss << " (launch_failed)";
    } else if (result.timed_out) {
        oss << " (timed_out)";
    } else {
        oss << " (exit_code=" << result.exit_code << ")";
    }
    if (!stdout_text.empty()) {
        oss << "\nstdout:\n" << stdout_text;
    }
    if (!stderr_text.empty()) {
        oss << "\nstderr:\n" << stderr_text;
    }
    if (!error_text.empty()) {
        oss << "\nerror:\n" << error_text;
    }
    return append_truncation_note(oss.str());
}

static std::string format_react_hard_memory_observation(const llama_cognitive_hard_memory_result & result) {
    json payload = {
        {"ok", result.result.ok},
        {"status_code", result.result.status_code},
        {"result_count", result.result.result_count},
        {"mean_similarity", result.result.retrieval_summary.mean_similarity},
        {"gain_support", result.result.retrieval_summary.gain_support},
        {"allostatic_support", result.result.retrieval_summary.allostatic_support},
    };
    if (result.result.error[0] != '\0') {
        payload["error"] = result.result.error;
    }
    return safe_json_to_str(payload);
}

static std::string format_react_codex_observation(const llama_codex_tool_result & result) {
    std::ostringstream oss;
    oss << "status=";
    if (result.launch_failed) {
        oss << "launch_failed";
    } else if (result.exit_code == 0) {
        oss << "ok";
    } else {
        oss << "error";
    }
    oss << "\nexit_code=" << result.exit_code;
    if (result.runtime_ms > 0) {
        oss << "\nruntime_ms=" << result.runtime_ms;
    }
    if (result.repo_changed) {
        oss << "\nrepo_changed=true";
    }
    if (result.rebuild_attempted) {
        oss << "\nrebuild=" << (result.rebuild_succeeded ? "ok" : "scheduled_or_failed");
    }
    if (result.accessibility_verified) {
        oss << "\naccessibility_verified=true";
    }
    const std::string summary = trim_ascii_copy(result.summary_text);
    const std::string manual = trim_ascii_copy(result.manual_requirements);
    const std::string changed = trim_ascii_copy(result.changed_files_excerpt);
    const std::string error = trim_ascii_copy(result.error_text);
    if (!summary.empty()) {
        oss << "\nsummary:\n" << summary;
    }
    if (!manual.empty()) {
        oss << "\nmanual_requirements:\n" << manual;
    }
    if (!changed.empty()) {
        oss << "\nchanged_files:\n" << changed;
    }
    if (!error.empty()) {
        oss << "\nerror:\n" << error;
    }
    return oss.str();
}

static bool parse_env_flag(const char * value, bool fallback) {
    if (!value) {
        return fallback;
    }
    return std::atoi(value) != 0;
}

static bool request_param_truthy(const std::string & value) {
    if (value.empty()) {
        return false;
    }

    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

static uint64_t parse_uint64_param(const std::string & value, uint64_t fallback = 0) {
    if (value.empty()) {
        return fallback;
    }
    try {
        return std::stoull(value);
    } catch (...) {
        return fallback;
    }
}

static std::string string_to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
    });
    return value;
}

static std::string shell_quote(const std::string & value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

static bool write_text_file(
        const std::filesystem::path & path,
        const std::string & text,
        std::string * out_error = nullptr) {
    try {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            if (out_error) {
                *out_error = "failed to open file for write";
            }
            return false;
        }
        out << text;
        return true;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = err.what();
        }
        return false;
    }
}

static std::string read_text_file_bounded(const std::filesystem::path & path, size_t cap) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::string text;
    text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (text.size() > cap) {
        text.resize(cap);
    }
    return text;
}

static bool run_permissive_bash_command(
        const std::string & working_directory,
        const std::string & command_text,
        int32_t timeout_ms,
        int32_t max_stdout_bytes,
        int32_t max_stderr_bytes,
        llama_bash_tool_result * out_result) {
    if (!out_result) {
        return false;
    }
    llama_bash_tool_request request = {};
    request.command_id = 1;
    request.tool_job_id = 1;
    request.timeout_ms = std::max(100, timeout_ms);
    request.cpu_time_limit_secs = std::max(1, timeout_ms / 1000 + 30);
    request.max_child_processes = 32;
    request.max_open_files = 128;
    request.max_file_size_bytes = 8 * 1024 * 1024;
    request.max_stdout_bytes = std::max(1, std::min(max_stdout_bytes, LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1));
    request.max_stderr_bytes = std::max(1, std::min(max_stderr_bytes, LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1));
    request.inherit_env = true;
    request.login_shell = false;
    request.reject_shell_metacharacters = false;
    request.command_ready = true;
    std::snprintf(request.bash_path, sizeof(request.bash_path), "%s", "/bin/bash");
    std::snprintf(request.working_directory, sizeof(request.working_directory), "%s", working_directory.c_str());
    std::snprintf(request.command_text, sizeof(request.command_text), "%s", command_text.c_str());
    request.allowed_commands[0] = '\0';
    request.blocked_patterns[0] = '\0';
    request.allowed_env[0] = '\0';
    return server_bash_tool_execute(request, out_result);
}

static std::string parse_manual_requirements_line(const std::string & text) {
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        const std::string trimmed = trim_ascii_copy(line);
        const std::string lower = string_to_lower(trimmed);
        if (lower.rfind("manual requirements:", 0) == 0) {
            return trim_ascii_copy(trimmed.substr(std::strlen("Manual requirements:")));
        }
    }
    return {};
}

static std::string remove_manual_requirements_line(const std::string & text) {
    std::istringstream input(text);
    std::ostringstream output;
    std::string line;
    bool first = true;
    while (std::getline(input, line)) {
        const std::string trimmed = trim_ascii_copy(line);
        if (string_to_lower(trimmed).rfind("manual requirements:", 0) == 0) {
            continue;
        }
        if (!first) {
            output << '\n';
        }
        output << line;
        first = false;
    }
    return trim_ascii_copy(output.str());
}

static std::string resolve_vicuna_core_system_prompt(const common_params & params_base) {
    const char * env_prompt = std::getenv("VICUNA_CORE_SYSTEM_PROMPT");
    if (env_prompt && env_prompt[0] != '\0') {
        const std::string trimmed = trim_ascii_copy(env_prompt);
        if (!trimmed.empty()) {
            return trimmed;
        }
    }

    const std::string configured = trim_ascii_copy(params_base.system_prompt);
    if (!configured.empty()) {
        return configured;
    }

    return trim_ascii_copy(llama_vicuna_core_system_prompt_default());
}

static std::string compose_vicuna_core_system_prefix(const std::string & core_prompt) {
    const std::string trimmed = trim_ascii_copy(core_prompt);
    if (trimmed.empty()) {
        return {};
    }
    return "System:\n" + trimmed + "\n\n";
}

static std::string proactive_mailbox_message(
        const llama_dmn_tick_trace & dmn_trace,
        const llama_remediation_plan & remediation,
        const llama_governance_trace & governance,
        float directness_preference,
        float verbosity_preference) {
    const bool terse = directness_preference >= 0.60f && verbosity_preference <= 0.45f;
    const std::string repair_message = trim_ascii_copy(bounded_cstr_to_string(governance.repair_message));
    if (!repair_message.empty()) {
        return repair_message;
    }

    auto concise_or_full = [terse](const std::string & concise, const std::string & full) {
        return terse ? concise : full;
    };

    if (governance.outcome == LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR || dmn_trace.pressure.repair >= 0.45f) {
        return concise_or_full(
                "I may need to correct course before I continue.",
                "I may need to correct course. I am reassessing before I continue.");
    }

    if (remediation.action == LLAMA_REMEDIATION_ACTION_GATHER_INFO) {
        switch (remediation.tool_kind == LLAMA_TOOL_KIND_NONE ? dmn_trace.tool_kind : remediation.tool_kind) {
            case LLAMA_TOOL_KIND_HARD_MEMORY_QUERY:
                return concise_or_full(
                        "I am checking related context before I continue.",
                        "I am checking related context before I continue so I do not miss something important.");
            case LLAMA_TOOL_KIND_BASH_CLI:
                return concise_or_full(
                        "I am inspecting the runtime state before I continue.",
                        "I am inspecting the runtime state before I continue so I can verify the next step.");
            default:
                return concise_or_full(
                        "I am checking one more thing before I continue.",
                        "I am checking one more thing before I continue so I can answer more reliably.");
        }
    }

    if (remediation.action == LLAMA_REMEDIATION_ACTION_ACTIVE_LORA_UPDATE) {
        return concise_or_full(
                "I am adjusting my approach before I continue.",
                "I am adjusting my approach before I continue so the next message is more useful.");
    }

    if (dmn_trace.pressure.uncertainty >= 0.45f) {
        return concise_or_full(
                "I want to verify a detail before I continue.",
                "I want to verify a detail before I continue rather than guess.");
    }

    if (dmn_trace.pressure.continuation >= 0.55f) {
        return concise_or_full(
                "I have a follow-up that should help.",
                "I have a short follow-up that should help before we move on.");
    }

    return concise_or_full(
            "I have a brief update before we continue.",
            "I have a brief update before we continue.");
}

static json build_proactive_response_object(
        const std::string & response_id,
        const std::string & message_id,
        const std::string & model_name,
        const std::string & text,
        std::time_t created_at) {
    const json content_part = {
        {"type",        "output_text"},
        {"annotations", json::array()},
        {"logprobs",    json::array()},
        {"text",        text},
    };
    return json {
        {"id",           response_id},
        {"object",       "response"},
        {"created_at",   created_at},
        {"completed_at", created_at},
        {"model",        model_name},
        {"status",       "completed"},
        {"output", json::array({
            json {
                {"type",    "message"},
                {"status",  "completed"},
                {"id",      message_id},
                {"role",    "assistant"},
                {"content", json::array({content_part})},
            }
        })},
        {"usage", json {
            {"input_tokens",  0},
            {"output_tokens", 0},
            {"total_tokens",  0},
        }},
    };
}

static std::vector<json> build_proactive_response_events(
        const std::string & response_id,
        const std::string & message_id,
        const std::string & text,
        const json & response_object) {
    const json partial_response = {
        {"id",         response_id},
        {"object",     "response"},
        {"created_at", response_object.at("created_at")},
        {"status",     "in_progress"},
    };
    const json added_item = {
        {"type",    "message"},
        {"status",  "in_progress"},
        {"id",      message_id},
        {"role",    "assistant"},
        {"content", json::array()},
    };
    const json content_part = {
        {"type",        "output_text"},
        {"annotations", json::array()},
        {"logprobs",    json::array()},
        {"text",        text},
    };

    return {
        json {
            {"event", "response.created"},
            {"data", {
                {"type", "response.created"},
                {"response", partial_response},
            }},
        },
        json {
            {"event", "response.in_progress"},
            {"data", {
                {"type", "response.in_progress"},
                {"response", partial_response},
            }},
        },
        json {
            {"event", "response.output_item.added"},
            {"data", {
                {"type",         "response.output_item.added"},
                {"output_index", 0},
                {"item",         added_item},
            }},
        },
        json {
            {"event", "response.content_part.added"},
            {"data", {
                {"type",          "response.content_part.added"},
                {"item_id",       message_id},
                {"output_index",  0},
                {"content_index", 0},
                {"part", json {
                    {"type", "output_text"},
                    {"text", ""},
                }},
            }},
        },
        json {
            {"event", "response.output_text.delta"},
            {"data", {
                {"type",          "response.output_text.delta"},
                {"item_id",       message_id},
                {"output_index",  0},
                {"content_index", 0},
                {"delta",         text},
            }},
        },
        json {
            {"event", "response.output_text.done"},
            {"data", {
                {"type",          "response.output_text.done"},
                {"item_id",       message_id},
                {"output_index",  0},
                {"content_index", 0},
                {"text",          text},
            }},
        },
        json {
            {"event", "response.content_part.done"},
            {"data", {
                {"type",          "response.content_part.done"},
                {"item_id",       message_id},
                {"output_index",  0},
                {"content_index", 0},
                {"part",          content_part},
            }},
        },
        json {
            {"event", "response.output_item.done"},
            {"data", {
                {"type",         "response.output_item.done"},
                {"output_index", 0},
                {"item", json {
                    {"type",    "message"},
                    {"status",  "completed"},
                    {"id",      message_id},
                    {"role",    "assistant"},
                    {"content", json::array({content_part})},
                }},
            }},
        },
        json {
            {"event", "response.completed"},
            {"data", {
                {"type",     "response.completed"},
                {"response", response_object},
            }},
        },
    };
}

static vicuna_mailbox_snapshot proactive_mailbox_snapshot_copy(const vicuna_proactive_mailbox & mailbox) {
    std::lock_guard<std::mutex> lock(mailbox.mutex);

    vicuna_mailbox_snapshot snapshot = {};
    snapshot.max_responses = mailbox.max_responses;
    snapshot.next_sequence_number = mailbox.next_sequence_number;
    snapshot.response_order = mailbox.response_order;
    snapshot.responses = mailbox.responses;
    snapshot.live_events = mailbox.live_events;
    snapshot.live_stream_connected = mailbox.live_stream_connected;
    snapshot.publish_total = mailbox.publish_total;
    snapshot.complete_total = mailbox.complete_total;
    snapshot.fail_total = mailbox.fail_total;
    snapshot.dropped_total = mailbox.dropped_total;
    snapshot.last_publish_ms = mailbox.last_publish_ms;
    return snapshot;
}

static json proactive_mailbox_to_json(const vicuna_proactive_mailbox & mailbox) {
    const vicuna_mailbox_snapshot snapshot = proactive_mailbox_snapshot_copy(mailbox);

    json responses = json::array();
    for (const std::string & response_id : snapshot.response_order) {
        auto it = snapshot.responses.find(response_id);
        if (it == snapshot.responses.end()) {
            continue;
        }

        json response_events = json::array();
        for (const auto & stored_event : it->second.events) {
            response_events.push_back({
                {"sequence_number", stored_event.sequence_number},
                {"event",           stored_event.event},
            });
        }

        responses.push_back({
            {"response_id",   it->second.response_id},
            {"created_ms",    it->second.created_ms},
            {"completed_ms",  it->second.completed_ms},
            {"response",      it->second.response},
            {"events",        response_events},
        });
    }

    return json {
        {"max_responses",         snapshot.max_responses},
        {"next_sequence_number",  snapshot.next_sequence_number},
        {"live_stream_connected", false},
        {"publish_total",         snapshot.publish_total},
        {"complete_total",        snapshot.complete_total},
        {"fail_total",            snapshot.fail_total},
        {"dropped_total",         snapshot.dropped_total},
        {"last_publish_ms",       snapshot.last_publish_ms},
        {"responses",             responses},
    };
}

static bool proactive_mailbox_from_json(const json & data, vicuna_proactive_mailbox * out_mailbox) {
    if (!out_mailbox || !data.is_object()) {
        return false;
    }

    vicuna_mailbox_snapshot restored = {};
    restored.max_responses = std::max<size_t>(1, json_value(data, "max_responses", restored.max_responses));
    restored.next_sequence_number = json_value(data, "next_sequence_number", restored.next_sequence_number);
    restored.publish_total = json_value(data, "publish_total", restored.publish_total);
    restored.complete_total = json_value(data, "complete_total", restored.complete_total);
    restored.fail_total = json_value(data, "fail_total", restored.fail_total);
    restored.dropped_total = json_value(data, "dropped_total", restored.dropped_total);
    restored.last_publish_ms = json_value(data, "last_publish_ms", restored.last_publish_ms);

    if (data.contains("responses") && data.at("responses").is_array()) {
        for (const auto & entry : data.at("responses")) {
            const std::string response_id = json_value(entry, "response_id", std::string());
            if (response_id.empty() || !entry.contains("response") || !entry.at("response").is_object()) {
                continue;
            }

            vicuna_stored_response stored = {};
            stored.response_id = response_id;
            stored.created_ms = json_value(entry, "created_ms", int64_t(0));
            stored.completed_ms = json_value(entry, "completed_ms", int64_t(0));
            stored.response = entry.at("response");

            if (entry.contains("events") && entry.at("events").is_array()) {
                for (const auto & event_entry : entry.at("events")) {
                    if (!event_entry.contains("event")) {
                        continue;
                    }
                    vicuna_mailbox_event stored_event = {};
                    stored_event.sequence_number = json_value(event_entry, "sequence_number", uint64_t(0));
                    stored_event.response_id = response_id;
                    stored_event.event = event_entry.at("event");
                    restored.live_events.push_back(stored_event);
                    stored.events.push_back(std::move(stored_event));
                }
            }

            restored.response_order.push_back(response_id);
            restored.responses.emplace(response_id, std::move(stored));
        }
    }

    {
        std::lock_guard<std::mutex> lock(out_mailbox->mutex);
        out_mailbox->max_responses = restored.max_responses;
        out_mailbox->next_sequence_number = restored.next_sequence_number;
        out_mailbox->response_order = std::move(restored.response_order);
        out_mailbox->responses = std::move(restored.responses);
        out_mailbox->live_events = std::move(restored.live_events);
        out_mailbox->live_stream_connected = false;
        out_mailbox->publish_total = restored.publish_total;
        out_mailbox->complete_total = restored.complete_total;
        out_mailbox->fail_total = restored.fail_total;
        out_mailbox->dropped_total = restored.dropped_total;
        out_mailbox->last_publish_ms = restored.last_publish_ms;
    }
    out_mailbox->cv.notify_all();
    return true;
}

static vicuna_telegram_outbox_snapshot telegram_outbox_snapshot_copy(const vicuna_telegram_outbox & outbox) {
    std::lock_guard<std::mutex> lock(outbox.mutex);

    vicuna_telegram_outbox_snapshot snapshot = {};
    snapshot.max_items = outbox.max_items;
    snapshot.next_sequence_number = outbox.next_sequence_number;
    snapshot.items = outbox.items;
    snapshot.publish_total = outbox.publish_total;
    snapshot.dropped_total = outbox.dropped_total;
    snapshot.last_publish_ms = outbox.last_publish_ms;
    return snapshot;
}

static json telegram_outbox_to_json(const vicuna_telegram_outbox & outbox) {
    const vicuna_telegram_outbox_snapshot snapshot = telegram_outbox_snapshot_copy(outbox);

    json items = json::array();
    for (const auto & item : snapshot.items) {
        items.push_back({
            {"sequence_number", item.sequence_number},
            {"kind", item.kind},
            {"approval_id", item.approval_id},
            {"chat_scope", item.chat_scope},
            {"dedupe_key", item.dedupe_key},
            {"text", item.text},
            {"reply_to_message_id", item.reply_to_message_id},
            {"question", item.question},
            {"options", item.options},
            {"created_at_ms", item.created_at_ms},
        });
    }

    return json{
        {"max_items", snapshot.max_items},
        {"next_sequence_number", snapshot.next_sequence_number},
        {"publish_total", snapshot.publish_total},
        {"dropped_total", snapshot.dropped_total},
        {"last_publish_ms", snapshot.last_publish_ms},
        {"items", items},
    };
}

static bool telegram_outbox_from_json(const json & data, vicuna_telegram_outbox * out_outbox) {
    if (!out_outbox || !data.is_object()) {
        return false;
    }

    vicuna_telegram_outbox_snapshot restored = {};
    restored.max_items = std::max<size_t>(1, json_value(data, "max_items", restored.max_items));
    restored.next_sequence_number = std::max<uint64_t>(1, json_value(data, "next_sequence_number", restored.next_sequence_number));
    restored.publish_total = json_value(data, "publish_total", restored.publish_total);
    restored.dropped_total = json_value(data, "dropped_total", restored.dropped_total);
    restored.last_publish_ms = json_value(data, "last_publish_ms", restored.last_publish_ms);

    if (data.contains("items") && data.at("items").is_array()) {
        uint64_t max_sequence = 0;
        for (const auto & entry : data.at("items")) {
            if (!entry.is_object()) {
                continue;
            }
            vicuna_telegram_outbox_item item = {};
            item.sequence_number = json_value(entry, "sequence_number", uint64_t(0));
            item.kind = trim_ascii_copy(json_value(entry, "kind", std::string()));
            item.approval_id = trim_ascii_copy(json_value(entry, "approval_id", std::string()));
            item.chat_scope = trim_ascii_copy(json_value(entry, "chat_scope", std::string()));
            item.dedupe_key = trim_ascii_copy(json_value(entry, "dedupe_key", std::string()));
            item.text = trim_ascii_copy(json_value(entry, "text", std::string()));
            item.reply_to_message_id = json_value(entry, "reply_to_message_id", int64_t(0));
            item.question = trim_ascii_copy(json_value(entry, "question", std::string()));
            item.created_at_ms = json_value(entry, "created_at_ms", int64_t(0));
            if (entry.contains("options") && entry.at("options").is_array()) {
                for (const auto & option : entry.at("options")) {
                    const std::string value = trim_ascii_copy(option.is_string() ? option.get<std::string>() : std::string());
                    if (!value.empty()) {
                        item.options.push_back(value);
                    }
                }
            }
            if (item.sequence_number == 0 || item.kind.empty() || item.chat_scope.empty()) {
                continue;
            }
            max_sequence = std::max(max_sequence, item.sequence_number);
            restored.items.push_back(std::move(item));
        }
        restored.next_sequence_number = std::max(restored.next_sequence_number, max_sequence + 1);
    }

    {
        std::lock_guard<std::mutex> lock(out_outbox->mutex);
        out_outbox->max_items = restored.max_items;
        out_outbox->next_sequence_number = restored.next_sequence_number;
        out_outbox->items = std::move(restored.items);
        out_outbox->publish_total = restored.publish_total;
        out_outbox->dropped_total = restored.dropped_total;
        out_outbox->last_publish_ms = restored.last_publish_ms;
    }

    return true;
}

static json bash_tool_config_to_json(const llama_bash_tool_config & config) {
    return json {
        {"enabled", config.enabled},
        {"inherit_env", config.inherit_env},
        {"login_shell", config.login_shell},
        {"reject_shell_metacharacters", config.reject_shell_metacharacters},
        {"timeout_ms", config.timeout_ms},
        {"cpu_time_limit_secs", config.cpu_time_limit_secs},
        {"max_child_processes", config.max_child_processes},
        {"max_open_files", config.max_open_files},
        {"max_file_size_bytes", config.max_file_size_bytes},
        {"max_stdout_bytes", config.max_stdout_bytes},
        {"max_stderr_bytes", config.max_stderr_bytes},
        {"bash_path", bounded_cstr_to_string(config.bash_path)},
        {"working_directory", bounded_cstr_to_string(config.working_directory)},
        {"allowed_commands", bounded_cstr_to_string(config.allowed_commands)},
        {"blocked_patterns", bounded_cstr_to_string(config.blocked_patterns)},
        {"allowed_env", bounded_cstr_to_string(config.allowed_env)},
    };
}

static std::string migrate_legacy_bash_allowed_commands(const std::string & allowed_commands) {
    static const std::string legacy_default =
            "pwd,ls,find,rg,cat,head,tail,grep,git,tavily-web-search,tools/openclaw-harness/bin/tavily-web-search";
    if (allowed_commands == legacy_default) {
        return {};
    }
    return allowed_commands;
}

static bool bash_tool_config_from_json(const json & data, llama_bash_tool_config * out_config) {
    if (!out_config || !data.is_object()) {
        return false;
    }

    *out_config = llama_bash_tool_default_config();
    out_config->enabled = json_value(data, "enabled", out_config->enabled);
    out_config->inherit_env = json_value(data, "inherit_env", out_config->inherit_env);
    out_config->login_shell = json_value(data, "login_shell", out_config->login_shell);
    out_config->reject_shell_metacharacters = json_value(data, "reject_shell_metacharacters", out_config->reject_shell_metacharacters);
    out_config->timeout_ms = json_value(data, "timeout_ms", out_config->timeout_ms);
    out_config->cpu_time_limit_secs = json_value(data, "cpu_time_limit_secs", out_config->cpu_time_limit_secs);
    out_config->max_child_processes = json_value(data, "max_child_processes", out_config->max_child_processes);
    out_config->max_open_files = json_value(data, "max_open_files", out_config->max_open_files);
    out_config->max_file_size_bytes = json_value(data, "max_file_size_bytes", out_config->max_file_size_bytes);
    out_config->max_stdout_bytes = json_value(data, "max_stdout_bytes", out_config->max_stdout_bytes);
    out_config->max_stderr_bytes = json_value(data, "max_stderr_bytes", out_config->max_stderr_bytes);
    std::snprintf(out_config->bash_path, sizeof(out_config->bash_path), "%s", json_value(data, "bash_path", std::string(out_config->bash_path)).c_str());
    std::snprintf(out_config->working_directory, sizeof(out_config->working_directory), "%s", json_value(data, "working_directory", std::string(out_config->working_directory)).c_str());
    const std::string allowed_commands = migrate_legacy_bash_allowed_commands(
            json_value(data, "allowed_commands", std::string(out_config->allowed_commands)));
    std::snprintf(out_config->allowed_commands, sizeof(out_config->allowed_commands), "%s", allowed_commands.c_str());
    std::snprintf(out_config->blocked_patterns, sizeof(out_config->blocked_patterns), "%s", json_value(data, "blocked_patterns", std::string(out_config->blocked_patterns)).c_str());
    std::snprintf(out_config->allowed_env, sizeof(out_config->allowed_env), "%s", json_value(data, "allowed_env", std::string(out_config->allowed_env)).c_str());
    return true;
}

static json hard_memory_config_to_json(const llama_hard_memory_config & config) {
    return json {
        {"enabled", config.enabled},
        {"archive_enabled", config.archive_enabled},
        {"include_profile_by_default", config.include_profile_by_default},
        {"archive_counterfactual_events", config.archive_counterfactual_events},
        {"timeout_ms", config.timeout_ms},
        {"max_results", config.max_results},
        {"query_threshold", config.query_threshold},
        {"archival_delta_threshold", config.archival_delta_threshold},
        {"base_url", bounded_cstr_to_string(config.base_url)},
        {"auth_token", bounded_cstr_to_string(config.auth_token)},
        {"container_tag", bounded_cstr_to_string(config.container_tag)},
        {"runtime_identity", bounded_cstr_to_string(config.runtime_identity)},
    };
}

static bool hard_memory_config_from_json(const json & data, llama_hard_memory_config * out_config) {
    if (!out_config || !data.is_object()) {
        return false;
    }

    *out_config = llama_hard_memory_default_config();
    out_config->enabled = json_value(data, "enabled", out_config->enabled);
    out_config->archive_enabled = json_value(data, "archive_enabled", out_config->archive_enabled);
    out_config->include_profile_by_default = json_value(data, "include_profile_by_default", out_config->include_profile_by_default);
    out_config->archive_counterfactual_events = json_value(data, "archive_counterfactual_events", out_config->archive_counterfactual_events);
    out_config->timeout_ms = json_value(data, "timeout_ms", out_config->timeout_ms);
    out_config->max_results = json_value(data, "max_results", out_config->max_results);
    out_config->query_threshold = json_value(data, "query_threshold", out_config->query_threshold);
    out_config->archival_delta_threshold = json_value(data, "archival_delta_threshold", out_config->archival_delta_threshold);
    std::snprintf(out_config->base_url, sizeof(out_config->base_url), "%s", json_value(data, "base_url", std::string(out_config->base_url)).c_str());
    std::snprintf(out_config->auth_token, sizeof(out_config->auth_token), "%s", json_value(data, "auth_token", std::string(out_config->auth_token)).c_str());
    std::snprintf(out_config->container_tag, sizeof(out_config->container_tag), "%s", json_value(data, "container_tag", std::string(out_config->container_tag)).c_str());
    std::snprintf(out_config->runtime_identity, sizeof(out_config->runtime_identity), "%s", json_value(data, "runtime_identity", std::string(out_config->runtime_identity)).c_str());
    return true;
}

static json model_extension_info_to_json(const llama_self_model_extension_info & info) {
    return json {
        {"source", info.source},
        {"source_tool_kind", info.source_tool_kind},
        {"kind", info.kind},
        {"domain", info.domain},
        {"lifecycle_stage", info.lifecycle_stage},
        {"flags", info.flags},
        {"support_count", info.support_count},
        {"value", info.value},
        {"desired_value", info.desired_value},
        {"desired_value_min", info.desired_value_min},
        {"desired_value_max", info.desired_value_max},
        {"confidence", info.confidence},
        {"salience", info.salience},
        {"gain_weight", info.gain_weight},
        {"allostatic_weight", info.allostatic_weight},
        {"surprise_score", info.surprise_score},
        {"relevance_score", info.relevance_score},
        {"admission_score", info.admission_score},
        {"permanence_score", info.permanence_score},
        {"stability_score", info.stability_score},
        {"allostatic_eligibility", info.allostatic_eligibility},
        {"key", bounded_cstr_to_string(info.key)},
        {"label", bounded_cstr_to_string(info.label)},
        {"content", bounded_cstr_to_string(info.content)},
    };
}

static bool model_extension_update_from_json(const json & data, llama_self_model_extension_update * out_update) {
    if (!out_update || !data.is_object()) {
        return false;
    }

    *out_update = llama_self_model_extension_default_update();
    out_update->source = json_value(data, "source", out_update->source);
    out_update->source_tool_kind = json_value(data, "source_tool_kind", out_update->source_tool_kind);
    out_update->kind = json_value(data, "kind", out_update->kind);
    out_update->domain = json_value(data, "domain", out_update->domain);
    out_update->lifecycle_stage = json_value(data, "lifecycle_stage", out_update->lifecycle_stage);
    out_update->flags = json_value(data, "flags", out_update->flags);
    out_update->support_count = json_value(data, "support_count", out_update->support_count);
    out_update->value = json_value(data, "value", out_update->value);
    out_update->desired_value = json_value(data, "desired_value", out_update->desired_value);
    out_update->desired_value_min = json_value(data, "desired_value_min", out_update->desired_value_min);
    out_update->desired_value_max = json_value(data, "desired_value_max", out_update->desired_value_max);
    out_update->confidence = json_value(data, "confidence", out_update->confidence);
    out_update->salience = json_value(data, "salience", out_update->salience);
    out_update->gain_weight = json_value(data, "gain_weight", out_update->gain_weight);
    out_update->allostatic_weight = json_value(data, "allostatic_weight", out_update->allostatic_weight);
    out_update->surprise_score = json_value(data, "surprise_score", out_update->surprise_score);
    out_update->relevance_score = json_value(data, "relevance_score", out_update->relevance_score);
    out_update->admission_score = json_value(data, "admission_score", out_update->admission_score);
    out_update->permanence_score = json_value(data, "permanence_score", out_update->permanence_score);
    out_update->stability_score = json_value(data, "stability_score", out_update->stability_score);
    out_update->allostatic_eligibility = json_value(data, "allostatic_eligibility", out_update->allostatic_eligibility);
    std::snprintf(out_update->key, sizeof(out_update->key), "%s", json_value(data, "key", std::string(out_update->key)).c_str());
    std::snprintf(out_update->label, sizeof(out_update->label), "%s", json_value(data, "label", std::string(out_update->label)).c_str());
    std::snprintf(out_update->content, sizeof(out_update->content), "%s", json_value(data, "content", std::string(out_update->content)).c_str());
    return true;
}


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;
    friend struct server_routes;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;

    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_webui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    server_state state = SERVER_STATE_LOADING_MODEL;
    bool sleeping = false;
    server_openclaw_fabric openclaw_fabric;
    vicuna_external_work_queue bash_work;
    vicuna_external_work_queue hard_memory_work;
    vicuna_external_work_queue codex_work;
    std::thread bash_worker;
    std::thread hard_memory_worker;
    std::thread codex_worker;
    mutable std::mutex runtime_state_mutex;
    std::unordered_map<int32_t, server_task> waiting_active_tasks;
    std::unordered_map<int32_t, server_task> waiting_dmn_tasks;
    std::unordered_map<std::string, vicuna_pending_approval> pending_approvals_by_id;
    std::unordered_map<int32_t, std::string> pending_approval_id_by_command;
    uint64_t next_pending_approval_ordinal = 1;
    std::unordered_map<int32_t, vicuna_foreground_consent_scope> foreground_consent_by_episode;
    uint64_t next_foreground_consent_scope_ordinal = 1;
    vicuna_runtime_persistence_state runtime_persistence;
    vicuna_provenance_repository_state provenance_repository;
    vicuna_external_observability external_observability;
    mutable vicuna_proactive_mailbox proactive_mailbox;
    mutable vicuna_telegram_outbox telegram_outbox;
    mutable vicuna_telegram_dialogue_history telegram_dialogue_history;
    mutable bool runtime_state_dirty = false;
    bool bash_tool_enabled = false;
    bool hard_memory_enabled = false;
    bool codex_tool_enabled = false;
    bool authoritative_react_control_enabled = false;
    bool active_loop_enabled = true;
    llama_bash_tool_config configured_bash_tool = {};
    llama_codex_tool_config configured_codex_tool = {};
    std::string core_system_prompt;
    std::string core_system_prompt_prefix;
    llama_tokens core_system_prompt_prefix_tokens;

    static int32_t count_pending_external_work(const vicuna_external_work_queue & queue) {
        std::lock_guard<std::mutex> lock(queue.mutex);
        return (int32_t) (queue.pending.size() + queue.inflight_ids.size());
    }

    bool has_waiting_active_tasks() const {
        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        return !waiting_active_tasks.empty();
    }

    struct authoritative_react_live_state {
        bool waiting_active_tasks = false;
        bool waiting_dmn_tasks = false;
        bool active_runner_live = false;
        bool dmn_runner_live = false;

        bool any_live() const {
            return waiting_active_tasks ||
                    waiting_dmn_tasks ||
                    active_runner_live ||
                    dmn_runner_live;
        }
    };

    authoritative_react_live_state collect_authoritative_react_live_state() const {
        authoritative_react_live_state state;
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            state.waiting_active_tasks = !waiting_active_tasks.empty();
            state.waiting_dmn_tasks = !waiting_dmn_tasks.empty();
        }

        if (!ctx || !authoritative_react_control_enabled) {
            return state;
        }

        llama_cognitive_active_runner_status active_runner = {};
        if (llama_cognitive_active_runner_get(ctx, &active_runner) == 0) {
            state.active_runner_live =
                    active_runner.active ||
                    active_runner.waiting_on_tool ||
                    active_runner.planning_active ||
                    active_runner.pending_command_id > 0;
        }

        llama_cognitive_dmn_runner_status dmn_runner = {};
        if (llama_cognitive_dmn_runner_get(ctx, &dmn_runner) == 0) {
            state.dmn_runner_live =
                    dmn_runner.active ||
                    dmn_runner.waiting_on_tool ||
                    dmn_runner.planning_active ||
                    dmn_runner.pending_command_id > 0;
        }

        return state;
    }

    static std::string default_execution_safety_class_for_tool_kind(int32_t tool_kind) {
        switch (tool_kind) {
            case LLAMA_TOOL_KIND_HARD_MEMORY_QUERY:
            case LLAMA_TOOL_KIND_TELEGRAM_ASK_OPTIONS:
                return "read_only";
            default:
                return "approval_required";
        }
    }

    static std::string execution_safety_class_for_side_effect_class(const std::string & side_effect_class) {
        std::string normalized = trim_ascii_copy(side_effect_class);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
            return (char) std::tolower(ch);
        });
        if (normalized == "memory_read" ||
            normalized == "network_read" ||
            normalized == "service_read") {
            return "read_only";
        }
        return normalized.empty() ? std::string() : "approval_required";
    }

    static bool task_requests_foreground_consent(const server_task & task) {
        return task.foreground_role == LLAMA_SELF_STATE_EVENT_USER &&
                !trim_ascii_copy(task.foreground_text).empty();
    }

    void register_foreground_consent_scope(server_task & task) {
        if (!task_requests_foreground_consent(task) ||
            !task.has_active_trace ||
            task.active_trace.episode_id <= 0) {
            return;
        }

        vicuna_foreground_consent_scope scope = {};
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            auto & stored = foreground_consent_by_episode[task.active_trace.episode_id];
            if (stored.scope_id.empty()) {
                stored.scope_id = "fgc_" + std::to_string(next_foreground_consent_scope_ordinal++);
                stored.created_at_ms = ggml_time_ms();
            }
            stored.task_id = task.id;
            stored.episode_id = task.active_trace.episode_id;
            stored.chat_scope = trim_ascii_copy(task.telegram_chat_scope);
            stored.telegram_message_id = std::max<int64_t>(0, task.telegram_message_id);
            stored.updated_at_ms = ggml_time_ms();
            scope = stored;
        }

        task.foreground_consent_requested = true;
        task.foreground_consent_active = true;
        task.foreground_consent_episode_id = scope.episode_id;
        task.foreground_consent_scope_id = scope.scope_id;
    }

    void expire_foreground_consent_scope(int32_t episode_id) {
        if (episode_id <= 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        foreground_consent_by_episode.erase(episode_id);
    }

    const vicuna_foreground_consent_scope * foreground_consent_scope_for_command_locked(
            const llama_cognitive_command & command) const {
        if (command.origin != LLAMA_COG_COMMAND_ORIGIN_ACTIVE || command.episode_id <= 0) {
            return nullptr;
        }
        auto it = foreground_consent_by_episode.find(command.episode_id);
        return it != foreground_consent_by_episode.end() ? &it->second : nullptr;
    }

    std::string command_execution_safety_class(
            const llama_cognitive_command & command,
            const server_openclaw_capability_runtime * capability) const {
        const std::string descriptor_class =
                capability ? trim_ascii_copy(capability->descriptor.execution_safety_class) : std::string();
        if (!descriptor_class.empty()) {
            return descriptor_class;
        }
        const std::string derived_class =
                capability ? execution_safety_class_for_side_effect_class(capability->descriptor.side_effect_class) : std::string();
        if (!derived_class.empty()) {
            return derived_class;
        }
        return default_execution_safety_class_for_tool_kind(command.tool_kind);
    }

    std::string command_authorization_source_locked(
            const llama_cognitive_command & command,
            const server_openclaw_capability_runtime * capability,
            const vicuna_pending_approval * approval = nullptr) const {
        const std::string safety_class = command_execution_safety_class(command, capability);
        if (trim_ascii_copy(safety_class) == "read_only") {
            return "read_only";
        }
        if (foreground_consent_scope_for_command_locked(command) != nullptr) {
            return "foreground_consent";
        }
        if (approval && approval->status == VICUNA_PENDING_APPROVAL_APPROVED) {
            return "explicit_approval";
        }
        return "approval_required";
    }

    bool command_requires_user_approval(
            const llama_cognitive_command & command,
            const server_openclaw_capability_runtime * capability) const {
        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        return command_authorization_source_locked(command, capability) == "approval_required";
    }

    const server_task * waiting_task_for_command_locked(const llama_cognitive_command & command) const {
        if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
            auto it = waiting_active_tasks.find(command.command_id);
            return it != waiting_active_tasks.end() ? &it->second : nullptr;
        }
        if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
            auto it = waiting_dmn_tasks.find(command.command_id);
            return it != waiting_dmn_tasks.end() ? &it->second : nullptr;
        }
        return nullptr;
    }

    std::string approval_chat_scope_locked(const llama_cognitive_command & command) const {
        const server_task * task = waiting_task_for_command_locked(command);
        if (task) {
            const std::string task_scope = trim_ascii_copy(task->telegram_chat_scope);
            if (!task_scope.empty()) {
                return task_scope;
            }
        }
        if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
            std::lock_guard<std::mutex> dialogue_lock(telegram_dialogue_history.mutex);
            return dmn_telegram_dialogue_scope_locked();
        }
        return {};
    }

    int64_t approval_reply_to_message_id_locked(const llama_cognitive_command & command) const {
        const server_task * task = waiting_task_for_command_locked(command);
        if (task && task->telegram_message_id > 0) {
            return task->telegram_message_id;
        }
        const std::string scope = approval_chat_scope_locked(command);
        if (scope.empty()) {
            return 0;
        }
        std::lock_guard<std::mutex> dialogue_lock(telegram_dialogue_history.mutex);
        auto it = telegram_dialogue_history.latest_message_id_by_scope.find(scope);
        return it != telegram_dialogue_history.latest_message_id_by_scope.end() ? it->second : 0;
    }

    std::string build_command_approval_question(
            const llama_cognitive_command & command,
            const server_openclaw_capability_runtime * capability) const {
        const std::string method_name =
                capability ? trim_ascii_copy(capability->descriptor.method_name) : std::string();
        const std::string family_name =
                capability ? trim_ascii_copy(capability->descriptor.tool_family_name) : std::string();
        const std::string capability_name =
                capability ? trim_ascii_copy(capability->descriptor.description) : std::string();

        if (command.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI) {
            llama_codex_tool_request request = {};
            if (llama_cognitive_codex_tool_get_request(ctx, command.command_id, &request) == 0) {
                const std::string task_text = trim_ascii_copy(request.intent_text);
                if (!task_text.empty()) {
                    return "Approve Codex to " + task_text + "?";
                }
            }
        }
        if (command.tool_kind == LLAMA_TOOL_KIND_BASH_CLI) {
            llama_bash_tool_request request = {};
            if (llama_cognitive_bash_tool_get_request(ctx, command.command_id, &request) == 0) {
                const std::string intent = trim_ascii_copy(request.intent_text);
                if (!intent.empty()) {
                    return "Approve running a bash command to " + intent + "?";
                }
            }
            return "Approve running a bash command?";
        }
        if (command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
            return "Approve writing to hard memory?";
        }
        if (command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY) {
            return "Approve sending a Telegram message?";
        }
        if (!family_name.empty() && !method_name.empty()) {
            return "Approve " + family_name + " " + method_name + "?";
        }
        if (!capability_name.empty()) {
            return "Approve " + capability_name + "?";
        }
        return "Approve this command?";
    }

    vicuna_pending_approval * pending_approval_for_command_locked(int32_t command_id) {
        auto id_it = pending_approval_id_by_command.find(command_id);
        if (id_it == pending_approval_id_by_command.end()) {
            return nullptr;
        }
        auto approval_it = pending_approvals_by_id.find(id_it->second);
        return approval_it != pending_approvals_by_id.end() ? &approval_it->second : nullptr;
    }

    const vicuna_pending_approval * pending_approval_for_command_locked(int32_t command_id) const {
        auto id_it = pending_approval_id_by_command.find(command_id);
        if (id_it == pending_approval_id_by_command.end()) {
            return nullptr;
        }
        auto approval_it = pending_approvals_by_id.find(id_it->second);
        return approval_it != pending_approvals_by_id.end() ? &approval_it->second : nullptr;
    }

    bool ensure_command_approval_prompt(
            const llama_cognitive_command & command,
            const server_openclaw_capability_runtime * capability,
            std::string * out_error) {
        const std::string chat_scope = approval_chat_scope_locked(command);
        if (chat_scope.empty()) {
            if (out_error) {
                *out_error = "approval required but no Telegram chat scope was available";
            }
            return false;
        }

        const int64_t reply_to_message_id = approval_reply_to_message_id_locked(command);
        const std::string question = build_command_approval_question(command, capability);
        const std::vector<std::string> options = {"allow", "disallow"};

        vicuna_pending_approval approval = {};
        bool created = false;
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            if (vicuna_pending_approval * existing = pending_approval_for_command_locked(command.command_id)) {
                return existing->status == VICUNA_PENDING_APPROVAL_PENDING_PUBLISH ||
                        existing->status == VICUNA_PENDING_APPROVAL_AWAITING_USER ||
                        existing->status == VICUNA_PENDING_APPROVAL_APPROVED;
            }

            approval.approval_id = "appr_" + std::to_string(next_pending_approval_ordinal++);
            approval.command_id = command.command_id;
            approval.origin = command.origin;
            approval.tool_kind = command.tool_kind;
            approval.tool_spec_index = command.tool_spec_index;
            approval.tool_job_id = command.tool_job_id;
            approval.capability_id = capability ? capability->descriptor.capability_id : trim_ascii_copy(command.capability_id);
            approval.tool_family_id = capability ? capability->descriptor.tool_family_id : std::string();
            approval.method_name = capability ? capability->descriptor.method_name : std::string();
            approval.chat_scope = chat_scope;
            approval.reply_to_message_id = reply_to_message_id;
            approval.question = question;
            approval.options = options;
            approval.dedupe_key = "approval-" + std::to_string(command.command_id);
            approval.status = VICUNA_PENDING_APPROVAL_PENDING_PUBLISH;
            approval.created_at_ms = ggml_time_ms();
            approval.updated_at_ms = approval.created_at_ms;
            pending_approval_id_by_command[command.command_id] = approval.approval_id;
            pending_approvals_by_id[approval.approval_id] = approval;
            created = true;
        }

        uint64_t sequence_number = 0;
        if (!telegram_outbox_publish_approval_request(
                    approval.approval_id,
                    approval.chat_scope,
                    approval.dedupe_key,
                    approval.question,
                    approval.options,
                    approval.reply_to_message_id,
                    &sequence_number)) {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            pending_approval_id_by_command.erase(command.command_id);
            pending_approvals_by_id.erase(approval.approval_id);
            if (out_error) {
                *out_error = "failed to enqueue approval prompt";
            }
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            auto it = pending_approvals_by_id.find(approval.approval_id);
            if (it != pending_approvals_by_id.end()) {
                it->second.status = VICUNA_PENDING_APPROVAL_AWAITING_USER;
                it->second.published_sequence_number = sequence_number;
                it->second.updated_at_ms = ggml_time_ms();
            }
        }

        if (created) {
            const json authorization = command_authorization_trace_json(command, capability);
            append_telegram_dialogue_assistant_turn(
                    approval.chat_scope,
                    approval.question,
                    command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN ? "telegram_approval_dmn" : "telegram_approval_active",
                    std::string(),
                    approval.dedupe_key);
            mark_runtime_state_dirty("approval-prompt");
            capture_tool_call_provenance(
                    "approval_request",
                    command,
                    pending_approval_to_json(approval),
                    &authorization);
        }
        return true;
    }

    void log_bash_tool_configuration(const llama_bash_tool_config & bash_tool) const {
        if (!bash_tool.enabled) {
            return;
        }
        SRV_INF("bash tool enabled: bash=%s cwd=%s timeout_ms=%d stdout=%d stderr=%d login=%d inherit_env=%d reject_meta=%d cpu_s=%d max_children=%d max_open_files=%d max_file_size=%d allow=%s block=%s env=%s\n",
                bash_tool.bash_path,
                bash_tool.working_directory[0] ? bash_tool.working_directory : ".",
                bash_tool.timeout_ms,
                bash_tool.max_stdout_bytes,
                bash_tool.max_stderr_bytes,
                bash_tool.login_shell ? 1 : 0,
                bash_tool.inherit_env ? 1 : 0,
                bash_tool.reject_shell_metacharacters ? 1 : 0,
                bash_tool.cpu_time_limit_secs,
                bash_tool.max_child_processes,
                bash_tool.max_open_files,
                bash_tool.max_file_size_bytes,
                bash_tool.allowed_commands,
                bash_tool.blocked_patterns,
                bash_tool.allowed_env);
    }

    bool configure_hard_memory_from_env(bool log_success) {
        llama_hard_memory_config hard_memory = llama_hard_memory_default_config();

        const char * hard_memory_url = getenv("VICUNA_HARD_MEMORY_URL");
        const char * hard_memory_token = getenv("VICUNA_HARD_MEMORY_TOKEN");
        const char * hard_memory_tag = getenv("VICUNA_HARD_MEMORY_CONTAINER_TAG");
        const char * hard_memory_runtime = getenv("VICUNA_HARD_MEMORY_RUNTIME_ID");
        const char * hard_memory_threshold = getenv("VICUNA_HARD_MEMORY_ARCHIVAL_DELTA_THRESHOLD");
        const char * hard_memory_query_threshold = getenv("VICUNA_HARD_MEMORY_QUERY_THRESHOLD");
        const char * hard_memory_timeout = getenv("VICUNA_HARD_MEMORY_TIMEOUT_MS");
        const char * hard_memory_results = getenv("VICUNA_HARD_MEMORY_MAX_RESULTS");

        if (hard_memory_url && hard_memory_token) {
            hard_memory.enabled = true;
            std::snprintf(hard_memory.base_url, sizeof(hard_memory.base_url), "%s", hard_memory_url);
            std::snprintf(hard_memory.auth_token, sizeof(hard_memory.auth_token), "%s", hard_memory_token);
            if (hard_memory_tag) {
                std::snprintf(hard_memory.container_tag, sizeof(hard_memory.container_tag), "%s", hard_memory_tag);
            }
            if (hard_memory_runtime) {
                std::snprintf(hard_memory.runtime_identity, sizeof(hard_memory.runtime_identity), "%s", hard_memory_runtime);
            } else {
                std::snprintf(hard_memory.runtime_identity, sizeof(hard_memory.runtime_identity), "%s", "vicuna-server");
            }
            if (hard_memory_threshold) {
                hard_memory.archival_delta_threshold = std::max(0.0f, (float) std::atof(hard_memory_threshold));
            }
            if (hard_memory_query_threshold) {
                hard_memory.query_threshold = std::max(0.0f, (float) std::atof(hard_memory_query_threshold));
            }
            if (hard_memory_timeout) {
                hard_memory.timeout_ms = std::max(100, std::atoi(hard_memory_timeout));
            }
            if (hard_memory_results) {
                hard_memory.max_results = std::max(1, std::atoi(hard_memory_results));
            }
        }

        if (llama_hard_memory_configure(ctx, hard_memory) != 0) {
            hard_memory_enabled = false;
            SRV_WRN("%s\n", "failed to configure hard memory");
            return false;
        }

        hard_memory_enabled = hard_memory.enabled;
        if (log_success && hard_memory.enabled) {
            SRV_INF("hard memory enabled: url=%s container=%s runtime=%s\n",
                    hard_memory.base_url,
                    hard_memory.container_tag,
                    hard_memory.runtime_identity);
        }
        return true;
    }

    bool configure_bash_tool_from_env(bool log_success) {
        llama_bash_tool_config bash_tool = llama_bash_tool_default_config();
        const char * bash_enabled = getenv("VICUNA_BASH_TOOL_ENABLED");
        const char * bash_path = getenv("VICUNA_BASH_TOOL_PATH");
        const char * bash_workdir = getenv("VICUNA_BASH_TOOL_WORKDIR");
        const char * bash_timeout = getenv("VICUNA_BASH_TOOL_TIMEOUT_MS");
        const char * bash_stdout = getenv("VICUNA_BASH_TOOL_MAX_STDOUT_BYTES");
        const char * bash_stderr = getenv("VICUNA_BASH_TOOL_MAX_STDERR_BYTES");
        const char * bash_login = getenv("VICUNA_BASH_TOOL_LOGIN_SHELL");
        const char * bash_inherit_env = getenv("VICUNA_BASH_TOOL_INHERIT_ENV");
        const char * bash_allowed_commands = getenv("VICUNA_BASH_TOOL_ALLOWED_COMMANDS");
        const char * bash_blocked_patterns = getenv("VICUNA_BASH_TOOL_BLOCKED_PATTERNS");
        const char * bash_allowed_env = getenv("VICUNA_BASH_TOOL_ALLOWED_ENV");
        const char * bash_reject_meta = getenv("VICUNA_BASH_TOOL_REJECT_SHELL_METACHARACTERS");
        const char * bash_cpu_secs = getenv("VICUNA_BASH_TOOL_CPU_TIME_LIMIT_SECS");
        const char * bash_max_children = getenv("VICUNA_BASH_TOOL_MAX_CHILD_PROCESSES");
        const char * bash_max_open_files = getenv("VICUNA_BASH_TOOL_MAX_OPEN_FILES");
        const char * bash_max_file_size = getenv("VICUNA_BASH_TOOL_MAX_FILE_SIZE_BYTES");

        if (bash_enabled) {
            bash_tool.enabled = atoi(bash_enabled) != 0;
        }
        if (bash_path) {
            std::snprintf(bash_tool.bash_path, sizeof(bash_tool.bash_path), "%s", bash_path);
        }
        if (bash_workdir) {
            std::snprintf(bash_tool.working_directory, sizeof(bash_tool.working_directory), "%s", bash_workdir);
        }
        if (bash_timeout) {
            bash_tool.timeout_ms = std::max(100, atoi(bash_timeout));
        }
        if (bash_stdout) {
            bash_tool.max_stdout_bytes = std::max(1, atoi(bash_stdout));
        }
        if (bash_stderr) {
            bash_tool.max_stderr_bytes = std::max(1, atoi(bash_stderr));
        }
        if (bash_login) {
            bash_tool.login_shell = atoi(bash_login) != 0;
        }
        if (bash_inherit_env) {
            bash_tool.inherit_env = atoi(bash_inherit_env) != 0;
        }
        if (bash_allowed_commands) {
            std::snprintf(bash_tool.allowed_commands, sizeof(bash_tool.allowed_commands), "%s", bash_allowed_commands);
        }
        if (bash_blocked_patterns) {
            std::snprintf(bash_tool.blocked_patterns, sizeof(bash_tool.blocked_patterns), "%s", bash_blocked_patterns);
        }
        if (bash_allowed_env) {
            std::snprintf(bash_tool.allowed_env, sizeof(bash_tool.allowed_env), "%s", bash_allowed_env);
        }
        if (bash_reject_meta) {
            bash_tool.reject_shell_metacharacters = parse_env_flag(bash_reject_meta, bash_tool.reject_shell_metacharacters);
        }
        if (bash_cpu_secs) {
            bash_tool.cpu_time_limit_secs = std::max(1, atoi(bash_cpu_secs));
        }
        if (bash_max_children) {
            bash_tool.max_child_processes = std::max(1, atoi(bash_max_children));
        }
        if (bash_max_open_files) {
            bash_tool.max_open_files = std::max(3, atoi(bash_max_open_files));
        }
        if (bash_max_file_size) {
            bash_tool.max_file_size_bytes = std::max(1024, atoi(bash_max_file_size));
        }

        if (llama_bash_tool_configure(ctx, &bash_tool) != 0) {
            bash_tool_enabled = false;
            SRV_WRN("%s\n", "failed to configure bash tool");
            return false;
        }

        configured_bash_tool = bash_tool;
        bash_tool_enabled = bash_tool.enabled;
        if (log_success) {
            log_bash_tool_configuration(bash_tool);
        }
        return true;
    }

    void reapply_env_external_tool_policy_after_restore() {
        const bool runtime_state_enabled = [&]() {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            return runtime_persistence.enabled;
        }();
        if (!runtime_state_enabled) {
            return;
        }

        const bool hard_memory_ok = configure_hard_memory_from_env(false);
        const bool bash_ok = configure_bash_tool_from_env(false);
        if (hard_memory_ok && bash_ok) {
            SRV_INF("reapplied env-derived external tool policy after runtime restore: bash=%d hard_memory=%d\n",
                    bash_tool_enabled ? 1 : 0,
                    hard_memory_enabled ? 1 : 0);
        } else {
            SRV_WRN("failed to reapply env-derived external tool policy after runtime restore: bash=%d hard_memory=%d\n",
                    bash_ok ? 1 : 0,
                    hard_memory_ok ? 1 : 0);
        }
    }

    llama_bash_tool_request make_bash_tool_request_from_config(
            int32_t command_id,
            int32_t origin,
            int32_t tool_job_id,
            const std::string & intent_text) const {
        llama_bash_tool_request request = {};
        request.command_id = command_id;
        request.origin = origin;
        request.tool_job_id = tool_job_id;
        request.timeout_ms = configured_bash_tool.timeout_ms;
        request.cpu_time_limit_secs = configured_bash_tool.cpu_time_limit_secs;
        request.max_child_processes = configured_bash_tool.max_child_processes;
        request.max_open_files = configured_bash_tool.max_open_files;
        request.max_file_size_bytes = configured_bash_tool.max_file_size_bytes;
        request.max_stdout_bytes = configured_bash_tool.max_stdout_bytes;
        request.max_stderr_bytes = configured_bash_tool.max_stderr_bytes;
        request.inherit_env = configured_bash_tool.inherit_env;
        request.login_shell = configured_bash_tool.login_shell;
        request.reject_shell_metacharacters = configured_bash_tool.reject_shell_metacharacters;
        std::snprintf(request.bash_path, sizeof(request.bash_path), "%s", configured_bash_tool.bash_path);
        std::snprintf(request.working_directory, sizeof(request.working_directory), "%s", configured_bash_tool.working_directory);
        std::snprintf(request.allowed_commands, sizeof(request.allowed_commands), "%s", configured_bash_tool.allowed_commands);
        std::snprintf(request.blocked_patterns, sizeof(request.blocked_patterns), "%s", configured_bash_tool.blocked_patterns);
        std::snprintf(request.allowed_env, sizeof(request.allowed_env), "%s", configured_bash_tool.allowed_env);
        std::snprintf(request.intent_text, sizeof(request.intent_text), "%s", intent_text.c_str());
        return request;
    }

    std::string resolve_openclaw_wrapper_command(
            const llama_bash_tool_request & request,
            const char * binary_name) const {
        const std::filesystem::path relative =
                std::filesystem::path("tools") / "openclaw-harness" / "bin" / binary_name;
        const std::string workdir = trim_ascii_copy(request.working_directory);
        if (!workdir.empty()) {
            const std::filesystem::path base(workdir);
            if (base.is_absolute()) {
                return (base / relative).lexically_normal().string();
            }
        }
        return relative.string();
    }

    std::string shared_context_origin_label(int32_t origin) const {
        switch (origin) {
            case LLAMA_SHARED_CONTEXT_ORIGIN_ACTIVE: return "active";
            case LLAMA_SHARED_CONTEXT_ORIGIN_DMN: return "dmn";
            case LLAMA_SHARED_CONTEXT_ORIGIN_TOOL: return "tool";
            default: return "system";
        }
    }

    std::string shared_context_kind_label(int32_t kind) const {
        switch (kind) {
            case LLAMA_SHARED_CONTEXT_KIND_USER_MESSAGE: return "user_message";
            case LLAMA_SHARED_CONTEXT_KIND_HIDDEN_THOUGHT: return "hidden_thought";
            case LLAMA_SHARED_CONTEXT_KIND_TOOL_CALL: return "tool_call";
            case LLAMA_SHARED_CONTEXT_KIND_TOOL_OBSERVATION: return "tool_observation";
            case LLAMA_SHARED_CONTEXT_KIND_VISIBLE_OUTPUT: return "visible_output";
            case LLAMA_SHARED_CONTEXT_KIND_EMOTIVE_MOMENT: return "emotive_moment";
            case LLAMA_SHARED_CONTEXT_KIND_CONTEXT_EVICTION: return "context_eviction";
            default: return "internal_summary";
        }
    }

    std::string shared_context_phase_label(int32_t phase) const {
        switch (phase) {
            case LLAMA_SHARED_CONTEXT_PHASE_THINK: return "think";
            case LLAMA_SHARED_CONTEXT_PHASE_SELECT_TOOL: return "select_tool";
            case LLAMA_SHARED_CONTEXT_PHASE_PREPARE_TOOL: return "prepare_tool";
            case LLAMA_SHARED_CONTEXT_PHASE_OBSERVE: return "observe";
            case LLAMA_SHARED_CONTEXT_PHASE_EMIT: return "emit";
            case LLAMA_SHARED_CONTEXT_PHASE_COMPRESS: return "compress";
            default: return "counterfactual";
        }
    }

    std::string current_emotive_moment_text() const {
        llama_emotive_moment_revision emotive = {};
        if (ctx) {
            (void) llama_self_state_refresh_time(ctx);
        }
        if (!ctx || llama_self_state_get_emotive_moment_revision(ctx, &emotive) != 0) {
            return {};
        }
        return trim_ascii_copy(emotive.text);
    }

    std::string current_emotive_style_directive() const {
        llama_emotive_moment_revision emotive = {};
        if (ctx) {
            (void) llama_self_state_refresh_time(ctx);
        }
        if (!ctx || llama_self_state_get_emotive_moment_revision(ctx, &emotive) != 0 || !emotive.valid) {
            return {};
        }

        const auto & affect = emotive.affective_vector;

        const char * tone = "steady";
        if (affect.arousal >= 0.72f || affect.cognitive_tension >= 0.62f) {
            tone = affect.valence >= 0.55f ? "energized" : "strained";
        } else if (affect.arousal <= 0.32f) {
            tone = affect.valence >= 0.55f ? "calm" : "subdued";
        } else if (affect.valence >= 0.68f) {
            tone = "warm";
        } else if (affect.valence <= 0.38f) {
            tone = "somber";
        }

        const char * stance = "balanced";
        if (affect.dominance >= 0.70f) {
            stance = affect.cognitive_tension >= 0.55f ? "firm" : "confident";
        } else if (affect.dominance <= 0.34f) {
            stance = "tentative";
        } else if (affect.cognitive_tension >= 0.58f) {
            stance = "careful";
        }

        const char * social_posture = "neutral";
        if (affect.social_closeness >= 0.68f) {
            social_posture = "affiliative";
        } else if (affect.social_closeness <= 0.30f) {
            social_posture = "reserved";
        } else if (affect.valence >= 0.62f) {
            social_posture = "receptive";
        }

        const char * verbosity = "moderately concise";
        if (affect.goal_pressure >= 0.70f || affect.cognitive_tension >= 0.64f) {
            verbosity = "brief and efficient";
        } else if (affect.arousal <= 0.30f && affect.goal_pressure <= 0.36f) {
            verbosity = "measured and slightly fuller";
        }

        std::ostringstream style;
        style << "Emotive style directive: let the current emotive state shape wording and pacing. "
              << "Use a " << tone << ", " << stance << ", " << social_posture
              << " tone; keep visible language " << verbosity << ". "
              << "Let this influence hidden reasoning and natural-language replies, "
              << "but do not let it alter tool XML, tool arguments, JSON, identifiers, or other structured output.";
        return style.str();
    }

    std::string current_emotive_prompt_context_text() const {
        const std::string emotive = current_emotive_moment_text();
        const std::string emotive_style = current_emotive_style_directive();
        if (emotive.empty() && emotive_style.empty()) {
            return {};
        }

        std::ostringstream out;
        if (!emotive.empty()) {
            out << "Current emotive moment: " << emotive << "\n";
        }
        if (!emotive_style.empty()) {
            out << emotive_style << "\n";
        }
        return trim_ascii_copy(out.str());
    }

    std::string shared_context_summary_text(int32_t limit = 8) const {
        if (!ctx) {
            return {};
        }

        llama_shared_cognitive_context_window window = {};
        const int32_t item_count = llama_shared_cognitive_context_count(ctx);
        if (item_count <= 0 || llama_shared_cognitive_context_get_window(ctx, &window) != 0) {
            return {};
        }

        std::ostringstream summary;
        summary << "Shared context window: items=" << window.item_count
                << ", tokens=" << window.token_count
                << ", head_revision=" << window.head_revision << ".";
        const int32_t start = std::max(0, item_count - std::max(1, limit));
        for (int32_t i = start; i < item_count; ++i) {
            llama_shared_cognitive_context_item item = {};
            if (llama_shared_cognitive_context_get_item(ctx, i, &item) != 0) {
                continue;
            }
            summary << "\n- [" << item.context_item_id << "] "
                    << shared_context_origin_label(item.origin) << ' '
                    << shared_context_kind_label(item.kind)
                    << " phase=" << shared_context_phase_label(item.phase)
                    << " source=" << item.episode_or_tick_id
                    << " plan=" << item.plan_id;
        }
        return summary.str();
    }

    bool should_include_shared_context_item_for_task(
            const server_task & task,
            const llama_shared_cognitive_context_item & item,
            bool use_telegram_dialogue) const {
        if (!use_telegram_dialogue) {
            return true;
        }

        if (item.kind == LLAMA_SHARED_CONTEXT_KIND_USER_MESSAGE ||
            item.kind == LLAMA_SHARED_CONTEXT_KIND_VISIBLE_OUTPUT) {
            return false;
        }

        const bool active_telegram_turn =
                task.react_origin == SERVER_REACT_ORIGIN_ACTIVE &&
                task.telegram_dialogue_active &&
                !normalize_telegram_chat_scope(task.telegram_chat_scope).empty();
        if (!active_telegram_turn) {
            return true;
        }

        if (!task.react_resuming_from_tool_result) {
            return false;
        }

        // During a resumed Telegram turn, the prompt must be able to see the just-admitted
        // non-visible shared-context chain even if the active trace was rehydrated under a
        // different episode id. Fresh user-facing turns still exclude these items entirely.
        return true;
    }

    std::string shared_context_summary_text_for_task(const server_task & task, int32_t limit = 8) const {
        if (!ctx) {
            return {};
        }

        const std::vector<common_chat_msg> telegram_dialogue = telegram_dialogue_messages_for_task(task);
        const bool use_telegram_dialogue = !telegram_dialogue.empty();

        llama_shared_cognitive_context_window window = {};
        const int32_t item_count = llama_shared_cognitive_context_count(ctx);
        if (item_count <= 0 || llama_shared_cognitive_context_get_window(ctx, &window) != 0) {
            return {};
        }

        std::vector<std::string> lines;
        const int32_t start = std::max(0, item_count - 24);
        for (int32_t i = start; i < item_count; ++i) {
            llama_shared_cognitive_context_item item = {};
            if (llama_shared_cognitive_context_get_item(ctx, i, &item) != 0) {
                continue;
            }
            if (!should_include_shared_context_item_for_task(task, item, use_telegram_dialogue)) {
                continue;
            }

            std::ostringstream line;
            line << "[" << item.context_item_id << "] "
                 << shared_context_origin_label(item.origin) << ' '
                 << shared_context_kind_label(item.kind)
                 << " phase=" << shared_context_phase_label(item.phase)
                 << " source=" << item.episode_or_tick_id
                 << " plan=" << item.plan_id;
            lines.push_back(line.str());
        }

        if (lines.empty()) {
            return {};
        }

        std::ostringstream summary;
        summary << "Shared context window: items=" << window.item_count
                << ", tokens=" << window.token_count
                << ", head_revision=" << window.head_revision << ".";
        const int32_t summary_start = std::max(0, (int32_t) lines.size() - std::max(1, limit));
        for (int32_t i = summary_start; i < (int32_t) lines.size(); ++i) {
            summary << "\n- " << lines[(size_t) i];
        }
        return summary.str();
    }

    bool shared_context_item_text(int32_t index, std::string * out_text) const {
        if (!ctx || !out_text) {
            return false;
        }

        const llama_token * tokens = nullptr;
        size_t n_tokens = 0;
        if (!ctx->self_state_trace_get_item_tokens(index, &tokens, &n_tokens)) {
            return false;
        }

        if (!tokens || n_tokens == 0) {
            out_text->clear();
            return true;
        }

        std::vector<llama_token> owned_tokens(tokens, tokens + n_tokens);
        *out_text = trim_ascii_copy(common_detokenize(vocab, owned_tokens, true));
        return true;
    }

    bool shared_context_item_to_chat_message(int32_t index, common_chat_msg * out_msg) const {
        if (!ctx || !out_msg) {
            return false;
        }

        llama_shared_cognitive_context_item item = {};
        if (llama_shared_cognitive_context_get_item(ctx, index, &item) != 0) {
            return false;
        }

        std::string content;
        if (!shared_context_item_text(index, &content) || content.empty()) {
            return false;
        }

        common_chat_msg msg;
        switch (item.kind) {
            case LLAMA_SHARED_CONTEXT_KIND_USER_MESSAGE:
                msg.role = item.role == LLAMA_SELF_STATE_EVENT_USER ? "user" : "assistant";
                break;
            case LLAMA_SHARED_CONTEXT_KIND_TOOL_OBSERVATION:
                msg.role = "tool";
                break;
            case LLAMA_SHARED_CONTEXT_KIND_HIDDEN_THOUGHT:
            case LLAMA_SHARED_CONTEXT_KIND_TOOL_CALL:
            case LLAMA_SHARED_CONTEXT_KIND_VISIBLE_OUTPUT:
            case LLAMA_SHARED_CONTEXT_KIND_INTERNAL_SUMMARY:
                msg.role = "assistant";
                break;
            default:
                return false;
        }
        msg.content = std::move(content);
        *out_msg = std::move(msg);
        return true;
    }

    std::vector<common_chat_msg> canonical_react_messages(
            const server_task & task,
            size_t history_start_index = 0) const {
        std::vector<common_chat_msg> messages;
        if (!ctx) {
            return messages;
        }

        const bool use_pinned_turn_state =
                task.react_origin == SERVER_REACT_ORIGIN_ACTIVE &&
                !task.react_last_step_name.empty();
        if (use_pinned_turn_state) {
            common_chat_msg user_msg;
            user_msg.role = "user";
            user_msg.content =
                    task.foreground_text.empty() ?
                            "Continue the active ReAct turn from the pinned turn state." :
                            task.foreground_text;
            messages.push_back(std::move(user_msg));

            std::vector<common_chat_msg> history_messages =
                    react_history_messages(task, history_start_index);
            if (!history_messages.empty()) {
                messages.insert(
                        messages.end(),
                        std::make_move_iterator(history_messages.begin()),
                        std::make_move_iterator(history_messages.end()));
            } else {
                common_chat_msg assistant_state;
                assistant_state.role = "assistant";
                assistant_state.content = "Last completed step: " + task.react_last_step_name;
                if (!task.react_last_step_output.empty()) {
                    assistant_state.content += "\nLast step output:\n" + task.react_last_step_output;
                }
                if (!task.react_last_step_result.empty()) {
                    assistant_state.content += "\nLast step result:\n" + task.react_last_step_result;
                }
                messages.push_back(std::move(assistant_state));

                if (!task.react_last_tool_observation.empty()) {
                    common_chat_msg tool_msg;
                    tool_msg.role = "tool";
                    tool_msg.content = task.react_last_tool_observation;
                    messages.push_back(std::move(tool_msg));
                }
            }

            return messages;
        }

        const std::vector<common_chat_msg> telegram_dialogue = telegram_dialogue_messages_for_task(task);
        const bool use_telegram_dialogue = !telegram_dialogue.empty();
        const int32_t item_count = llama_shared_cognitive_context_count(ctx);
        const int32_t start = std::max(0, item_count - 24);
        messages.reserve(messages.size() + (size_t) std::max(1, item_count - start));
        for (int32_t i = start; i < item_count; ++i) {
            llama_shared_cognitive_context_item item = {};
            if (llama_shared_cognitive_context_get_item(ctx, i, &item) != 0) {
                continue;
            }
            if (!should_include_shared_context_item_for_task(task, item, use_telegram_dialogue)) {
                continue;
            }
            common_chat_msg msg;
            if (shared_context_item_to_chat_message(i, &msg)) {
                messages.push_back(std::move(msg));
            }
        }

        if (use_telegram_dialogue) {
            messages.insert(messages.end(), telegram_dialogue.begin(), telegram_dialogue.end());
        }

        if (messages.empty()) {
            common_chat_msg seed;
            seed.role = "user";
            seed.content =
                    task.react_origin == SERVER_REACT_ORIGIN_DMN ?
                    "Continue one hidden DMN reasoning step from the canonical shared cognitive context." :
                    "Continue the active ReAct turn from the canonical shared cognitive context.";
            messages.push_back(std::move(seed));
        }

        return messages;
    }

    static std::string lowercase_ascii_copy_local(const std::string & text) {
        std::string lowered = text;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
            return (char) std::tolower(ch);
        });
        return lowered;
    }

    static std::vector<std::string> foreground_grounding_terms(const std::string & text) {
        static const std::unordered_set<std::string> stop_words = {
            "what", "when", "where", "which", "will", "would", "should", "could", "about",
            "there", "their", "them", "this", "that", "with", "from", "your", "have",
            "does", "than", "just", "into", "tell", "like", "supposed", "please", "right",
            "current", "latest", "today", "tomorrow", "yesterday",
        };

        std::string normalized = lowercase_ascii_copy_local(text);
        for (char & ch : normalized) {
            if (!std::isalnum((unsigned char) ch)) {
                ch = ' ';
            }
        }

        std::istringstream in(normalized);
        std::vector<std::string> terms;
        std::unordered_set<std::string> seen;
        std::string token;
        while (in >> token) {
            if (token.size() < 4 || stop_words.count(token) > 0 || seen.count(token) > 0) {
                continue;
            }
            seen.insert(token);
            terms.push_back(token);
            if (terms.size() >= 8) {
                break;
            }
        }
        return terms;
    }

    static bool text_has_fact_like_payload(const std::string & text) {
        bool has_digit = false;
        for (char ch : text) {
            if (std::isdigit((unsigned char) ch)) {
                has_digit = true;
                break;
            }
        }
        if (has_digit) {
            return true;
        }

        const std::string lowered = lowercase_ascii_copy_local(text);
        return lowered.find('$') != std::string::npos ||
               lowered.find("fahrenheit") != std::string::npos ||
               lowered.find("celsius") != std::string::npos ||
               lowered.find("degrees") != std::string::npos ||
               lowered.find("forecast") != std::string::npos;
    }

    bool canonical_context_has_grounded_answer_candidate(const server_task & task) const {
        if (task.foreground_text.empty()) {
            return false;
        }

        const bool fresh_mutable_telegram_turn =
                task.react_origin == SERVER_REACT_ORIGIN_ACTIVE &&
                task.foreground_role == LLAMA_SELF_STATE_EVENT_USER &&
                task.telegram_dialogue_active &&
                !task.react_resuming_from_tool_result &&
                foreground_request_requires_fresh_tool_grounding(task.foreground_text);
        if (fresh_mutable_telegram_turn) {
            return false;
        }

        const std::vector<std::string> terms = foreground_grounding_terms(task.foreground_text);
        if (terms.empty()) {
            return false;
        }

        const std::vector<common_chat_msg> messages = canonical_react_messages(task);
        for (const auto & msg : messages) {
            if (msg.role == "user" || msg.role == "system") {
                continue;
            }

            const std::string lowered = lowercase_ascii_copy_local(trim_ascii_copy(msg.content));
            if (lowered.empty()) {
                continue;
            }

            int overlap = 0;
            for (const std::string & term : terms) {
                if (lowered.find(term) != std::string::npos) {
                    overlap++;
                }
            }

            if (overlap >= 2 && text_has_fact_like_payload(lowered)) {
                return true;
            }
        }

        return false;
    }

    bool react_turn_requires_retry_tool_escalation(const server_task & task) const {
        if (task.react_origin != SERVER_REACT_ORIGIN_ACTIVE ||
                task.foreground_role != LLAMA_SELF_STATE_EVENT_USER ||
                task.react_resuming_from_tool_result ||
                !react_has_available_tools(task)) {
            return false;
        }

        return authoritative_retry_requires_tool_escalation(
                task.foreground_text,
                task.react_retry_count);
    }

    bool canonical_context_has_direct_tool_answer_observation(const server_task & task) const {
        const std::vector<common_chat_msg> messages = canonical_react_messages(task);
        for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
            if (it->role != "tool") {
                continue;
            }
            const std::string lowered = lowercase_ascii_copy_local(trim_ascii_copy(it->content));
            if (lowered.find("\"answer\":") != std::string::npos ||
                    lowered.find("\nanswer:") != std::string::npos ||
                    lowered.find("\nanswer=") != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    bool latest_tool_observation_indicates_failure(const server_task & task) const {
        const std::string observation =
                !task.react_last_tool_observation.empty() ?
                        task.react_last_tool_observation :
                        [&]() -> std::string {
                            const std::vector<common_chat_msg> messages = canonical_react_messages(task);
                            for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
                                if (it->role == "tool" && !trim_ascii_copy(it->content).empty()) {
                                    return it->content;
                                }
                            }
                            return {};
                        }();
        const std::string lowered = lowercase_ascii_copy_local(trim_ascii_copy(observation));
        if (lowered.empty()) {
            return false;
        }
        return lowered.find("status=error") != std::string::npos ||
               lowered.find("status=launch_failed") != std::string::npos ||
               lowered.find("status=timed_out") != std::string::npos ||
               lowered.find("\"ok\": false") != std::string::npos ||
               lowered.find("\nerror:") != std::string::npos ||
               lowered.find("\nstderr:") != std::string::npos;
    }

    const server_openclaw_capability_runtime * react_capability_by_id(const std::string & capability_id) const {
        if (capability_id.empty()) {
            return nullptr;
        }
        const auto & capabilities = openclaw_fabric.capabilities();
        for (const auto & capability : capabilities) {
            if (capability.descriptor.capability_id == capability_id) {
                return &capability;
            }
        }
        return nullptr;
    }

    bool latest_tool_step_was_read_only(const server_task & task) const {
        const server_openclaw_capability_runtime * capability =
                react_capability_by_id(trim_ascii_copy(task.react_last_tool_capability_id));
        if (!capability) {
            return false;
        }

        const std::string side_effect_class =
                lowercase_ascii_copy_local(trim_ascii_copy(capability->descriptor.side_effect_class));
        if (side_effect_class == "memory_read" || side_effect_class == "network_read" || side_effect_class == "service_read") {
            return true;
        }

        return (capability->descriptor.tool_flags & LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT) == 0;
    }

    static bool response_text_claims_external_inaccessibility(const std::string & text) {
        const std::string lowered = lowercase_ascii_copy_local(trim_ascii_copy(text));
        if (lowered.empty()) {
            return false;
        }
        return lowered.find("unable to access") != std::string::npos ||
               lowered.find("cannot access") != std::string::npos ||
               lowered.find("can't access") != std::string::npos ||
               lowered.find("do not have access") != std::string::npos ||
               lowered.find("don't have access") != std::string::npos ||
               lowered.find("cannot interact with external") != std::string::npos ||
               lowered.find("can't interact with external") != std::string::npos ||
               lowered.find("external services like") != std::string::npos;
    }

    bool react_turn_requires_first_tool_call(const server_task & task) const {
        if (task.react_origin != SERVER_REACT_ORIGIN_ACTIVE ||
                task.foreground_role != LLAMA_SELF_STATE_EVENT_USER ||
                task.react_resuming_from_tool_result ||
                !react_has_available_tools(task)) {
            return false;
        }

        if (!foreground_request_requires_fresh_tool_grounding(task.foreground_text)) {
            return false;
        }

        if (react_turn_requires_retry_tool_escalation(task)) {
            return true;
        }

        return !canonical_context_has_grounded_answer_candidate(task);
    }

    void apply_core_system_prompt_prefix(server_task & task) const {
        const std::string emotive_context = current_emotive_prompt_context_text();
        if (core_system_prompt_prefix_tokens.empty() && emotive_context.empty()) {
            return;
        }

        llama_tokens prefix_tokens = core_system_prompt_prefix_tokens;
        if (!emotive_context.empty()) {
            const std::string runtime_prefix = std::string("System:\n") + emotive_context + "\n\n";
            const llama_tokens emotive_tokens = common_tokenize(vocab, runtime_prefix, false, true);
            prefix_tokens.insert(prefix_tokens.end(), emotive_tokens.begin(), emotive_tokens.end());
        }

        server_tokens prefixed(prefix_tokens, task.tokens.has_mtmd);
        prefixed.push_back(task.tokens);
        task.tokens = std::move(prefixed);
        task.params.n_keep = std::max(task.params.n_keep, (int32_t) prefix_tokens.size());
    }

    std::vector<int32_t> react_available_spec_indexes(const server_task & task) const {
        std::vector<int32_t> spec_indexes;
        if (!ctx) {
            return spec_indexes;
        }

        const bool want_dmn = task.react_origin == SERVER_REACT_ORIGIN_DMN;
        for (int32_t i = 0, n = llama_cognitive_tool_spec_count(ctx); i < n; ++i) {
            llama_cognitive_tool_spec spec = {};
            if (llama_cognitive_tool_spec_get(ctx, i, &spec) != 0) {
                continue;
            }
            const bool eligible = want_dmn ?
                    (spec.flags & LLAMA_COG_TOOL_DMN_ELIGIBLE) != 0 :
                    (spec.flags & LLAMA_COG_TOOL_ACTIVE_ELIGIBLE) != 0;
            if (eligible) {
                spec_indexes.push_back(i);
            }
        }
        return spec_indexes;
    }

    bool react_spec_index_exposed(const server_task & task, int32_t spec_index) const {
        const std::vector<int32_t> available = react_available_spec_indexes(task);
        return std::find(available.begin(), available.end(), spec_index) != available.end();
    }

    bool react_has_available_tools(const server_task & task) const {
        return !react_available_spec_indexes(task).empty();
    }

    const char * react_tool_stage_name(int32_t stage) const {
        switch (stage) {
            case SERVER_REACT_TOOL_STAGE_SELECT_TOOL:
                return "select_tool_family";
            case SERVER_REACT_TOOL_STAGE_SELECT_METHOD:
                return "select_method";
            case SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL:
                return "emit_arguments";
            case SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL:
                return "decide_after_tool";
            case SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE:
                return "emit_response";
            default:
                return "none";
        }
    }

    static const char * react_terminal_action_label(int32_t action) {
        switch (action) {
            case LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER:
                return "answer";
            case LLAMA_AUTHORITATIVE_REACT_ACTION_ASK:
                return "ask";
            case LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT:
                return "wait";
            case LLAMA_AUTHORITATIVE_REACT_ACTION_ACT:
                return "act";
            default:
                return "";
        }
    }

    json react_selector_schema(
            const char * action_value,
            const char * key,
            const char * description,
            const std::vector<std::string> & values) const {
        json enum_values = json::array();
        for (const auto & value : values) {
            const std::string trimmed = trim_ascii_copy(value);
            if (!trimmed.empty()) {
                enum_values.push_back(trimmed);
            }
        }
        return json{
            {"type", "object"},
            {"additionalProperties", false},
            {"required", json::array({"action", key})},
            {"properties", json{
                {"action", json{
                    {"type", "string"},
                    {"enum", json::array({action_value ? action_value : ""})},
                    {"description", "The controller action required for this staged phase."}
                }},
                {key, json{
                    {"type", "string"},
                    {"enum", enum_values},
                    {"description", description ? description : ""}
                }}
            }}
        };
    }

    static json react_wrap_method_argument_schema(
            const json & input_schema,
            const std::string & action_value) {
        json wrapped = input_schema;
        if (!wrapped.is_object()) {
            wrapped = json{
                    {"type", "object"},
                    {"additionalProperties", false},
                    {"properties", json::object()},
            };
        }
        wrapped["type"] = "object";
        if (!wrapped.contains("properties") || !wrapped["properties"].is_object()) {
            wrapped["properties"] = json::object();
        }
        wrapped["properties"]["action"] = json{
                {"type", "string"},
                {"enum", json::array({action_value})},
                {"description", "The controller action required for this staged phase."}
        };
        json required = json::array();
        if (wrapped.contains("required") && wrapped["required"].is_array()) {
            required = wrapped["required"];
        }
        bool have_action = false;
        for (const auto & entry : required) {
            if (entry.is_string() && entry.get<std::string>() == "action") {
                have_action = true;
                break;
            }
        }
        if (!have_action) {
            required.push_back("action");
        }
        wrapped["required"] = required;
        wrapped["additionalProperties"] = false;
        return wrapped;
    }

    static bool react_parse_visible_json_object(
            const std::string & visible_text,
            json * out_value,
            std::string * out_error) {
        if (!out_value) {
            if (out_error) {
                *out_error = "missing output json target";
            }
            return false;
        }

        const std::string trimmed = trim_ascii_copy(visible_text);
        if (trimmed.empty()) {
            if (out_error) {
                *out_error = "controller stage emitted no visible JSON payload";
            }
            return false;
        }

        try {
            *out_value = json::parse(trimmed);
        } catch (const std::exception & e) {
            if (out_error) {
                *out_error = std::string("controller stage emitted invalid JSON: ") + e.what();
            }
            return false;
        }

        if (!out_value->is_object()) {
            if (out_error) {
                *out_error = "controller stage must emit a single JSON object";
            }
            return false;
        }

        return true;
    }

    static bool react_parse_post_tool_decision_json(
            const std::string & visible_text,
            int32_t * out_action,
            std::string * out_error) {
        json payload;
        if (!react_parse_visible_json_object(visible_text, &payload, out_error)) {
            return false;
        }
        const std::string stage_action = trim_ascii_copy(payload.value("action", std::string()));
        if (stage_action != "decide") {
            if (out_error) {
                *out_error = "post-tool decision stage must emit action=\"decide\"";
            }
            return false;
        }
        if (payload.size() != 2 || !payload.contains("decision") || !payload["decision"].is_string()) {
            if (out_error) {
                *out_error = "post-tool decision stage must emit only {\"action\":\"decide\",\"decision\":\"answer|ask|select_tool|wait\"}";
            }
            return false;
        }

        std::string decision = trim_ascii_copy(payload["decision"].get<std::string>());
        std::transform(decision.begin(), decision.end(), decision.begin(), [](unsigned char ch) {
            if (ch == '-' || ch == ' ') {
                return '_';
            }
            return (char) std::tolower(ch);
        });
        int32_t action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
        if (decision == "select_tool") {
            action = LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
        } else if (decision == "answer") {
            action = LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER;
        } else if (decision == "ask") {
            action = LLAMA_AUTHORITATIVE_REACT_ACTION_ASK;
        } else if (decision == "wait") {
            action = LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT;
        } else {
            if (out_error) {
                *out_error = "post-tool decision must be one of answer, ask, select_tool, or wait";
            }
            return false;
        }

        if (out_action) {
            *out_action = action;
        }
        return true;
    }

    static bool react_parse_terminal_response_json(
            const std::string & visible_text,
            const std::string & expected_action,
            std::string * out_response_text,
            std::string * out_error) {
        json payload;
        if (!react_parse_visible_json_object(visible_text, &payload, out_error)) {
            return false;
        }
        const std::string stage_action = trim_ascii_copy(payload.value("action", std::string()));
        if (stage_action != expected_action) {
            if (out_error) {
                *out_error = std::string("response stage must emit action=\"") + expected_action + "\"";
            }
            return false;
        }
        if (payload.size() != 2 || !payload.contains("assistant_text") || !payload["assistant_text"].is_string()) {
            if (out_error) {
                *out_error = std::string("response stage must emit only {\"action\":\"") + expected_action + "\",\"assistant_text\":\"...\"}";
            }
            return false;
        }

        const std::string assistant_text = trim_ascii_copy(payload["assistant_text"].get<std::string>());
        if (assistant_text.empty()) {
            if (out_error) {
                *out_error = "response stage assistant_text must not be empty";
            }
            return false;
        }

        auto ends_with_terminal_punctuation = [](const std::string & text) {
            for (auto it = text.rbegin(); it != text.rend(); ++it) {
                const unsigned char ch = (unsigned char) *it;
                if (std::isspace(ch)) {
                    continue;
                }
                if (ch == '"' || ch == '\'' || ch == ')' || ch == ']' || ch == '}') {
                    continue;
                }
                return ch == '.' || ch == '!' || ch == '?';
            }
            return false;
        };

        if (!ends_with_terminal_punctuation(assistant_text)) {
            if (out_error) {
                *out_error = "response stage assistant_text must be a complete sentence ending with terminal punctuation";
            }
            return false;
        }

        if (out_response_text) {
            *out_response_text = assistant_text;
        }
        return true;
    }

    std::string react_assistant_prefill_for_task(const server_task & task) const {
        const bool first_tool_call_required = react_turn_requires_first_tool_call(task);
        const bool tool_stage_select =
                task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_TOOL ||
                (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_NONE && first_tool_call_required);

        if (tool_stage_select) {
            return std::string(
                    "<think>\n"
                    "Thought: select the appropriate tool based on context and the original request.\n"
                    "Action: select_tool\n"
                    "</think>\n");
        }

        if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_METHOD &&
                !task.react_selected_tool_family_id.empty()) {
            return std::string(
                    "<think>\n"
                    "Thought: select the appropriate method based on context and the original request.\n"
                    "Action: select_method\n"
                    "</think>\n");
        }

        if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL &&
                !task.react_selected_tool_family_id.empty() &&
                !task.react_selected_method_name.empty()) {
            return std::string(
                    "<think>\n"
                    "Thought: create the appropriate payload based on context and the original request.\n"
                    "Action: act\n"
                    "</think>\n");
        }

        if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL) {
            return std::string(
                    "<think>\n"
                    "Thought: select the appropriate next step based on the completed tool observation, context, and original request.\n"
                    "Action: decide\n"
                    "</think>\n");
        }

        if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE &&
                task.react_pending_terminal_action != LLAMA_AUTHORITATIVE_REACT_ACTION_NONE) {
            return std::string(
                    "<think>\n"
                    "Thought: write the appropriate response based on the completed tool observation, context, and original request.\n"
                    "Action: ") + react_terminal_action_label(task.react_pending_terminal_action) + "\n</think>\n";
        }

        return std::string(
                "<think>\n"
                "Thought: follow the authoritative turn policy based on the current context.\n"
                "Action: ");
    }

    bool apply_chat_prompt_to_task(server_task & task, const common_chat_params & chat_completion) {
        std::vector<server_tokens> prompts = tokenize_input_prompts(
                vocab,
                nullptr,
                json(chat_completion.prompt),
                true,
                true);
        if (prompts.empty()) {
            return false;
        }

        task.tokens = std::move(prompts.front());
        apply_core_system_prompt_prefix(task);
        task.params.sampling.grammar = chat_completion.grammar;
        task.params.sampling.grammar_lazy = chat_completion.grammar_lazy;
        task.params.sampling.grammar_triggers = chat_completion.grammar_triggers;
        task.params.sampling.preserved_tokens.clear();
        for (const auto & preserved : chat_completion.preserved_tokens) {
            const auto ids = common_tokenize(vocab, preserved, false, true);
            if (ids.size() == 1) {
                task.params.sampling.preserved_tokens.insert(ids[0]);
            }
        }

        task.params.chat_parser_params.format = chat_completion.format;
        task.params.chat_parser_params.parser = {};
        if (!chat_completion.parser.empty()) {
            task.params.chat_parser_params.parser.load(chat_completion.parser);
        }
        task.params.chat_parser_params.thinking_forced_open = chat_completion.thinking_forced_open;
        task.params.chat_parser_params.parse_tool_calls = false;
        return true;
    }

    bool prepare_react_prompt(server_task & task) {
        if (!server_task_should_prepare_authoritative_react(task)) {
            return false;
        }

        task.react_assistant_prefill = react_assistant_prefill_for_task(task);
        task.react_tools.clear();
        std::string staged_guidance;
        const std::vector<int32_t> available_spec_indexes = react_available_spec_indexes(task);
        if (!available_spec_indexes.empty()) {
            if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_METHOD) {
                if (!openclaw_fabric.render_tool_method_selection_guidance(
                            task.react_selected_tool_family_id,
                            &staged_guidance,
                            &available_spec_indexes)) {
                    return false;
                }
            } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL) {
                if (!openclaw_fabric.render_tool_argument_json_guidance(
                            task.react_selected_tool_family_id,
                            task.react_selected_method_name,
                            &staged_guidance,
                            &available_spec_indexes)) {
                    return false;
                }
            } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL) {
                staged_guidance =
                        "Available next-step decisions after this tool result:\n"
                        "- answer: the completed tool observation and admitted context are enough to answer now.\n"
                        "- ask: a user-facing clarification question is required before more work.\n"
                        "- select_tool: another tool invocation is still required.\n"
                        "- wait: only if another already-issued external tool is still pending.\n";
            } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE) {
                const std::string expected_action = react_terminal_action_label(task.react_pending_terminal_action);
                staged_guidance =
                        std::string("Emit exactly one JSON object in the shape {\"action\":\"") +
                        (expected_action.empty() ? "answer_or_ask" : expected_action) +
                        "\",\"assistant_text\":\"...\"}. "
                        "The assistant_text value must contain only the final plain-prose user-facing " +
                        (task.react_pending_terminal_action == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK ?
                                "question" :
                                "response") +
                        " for this step.";
            } else if (!openclaw_fabric.render_tool_family_selection_guidance(
                               &staged_guidance,
                               &available_spec_indexes)) {
                return false;
            }
        }
        common_chat_msg phase_system;
        phase_system.role = "system";
        const std::string emotive_context = current_emotive_prompt_context_text();
        const bool use_pinned_turn_state =
                task.react_origin == SERVER_REACT_ORIGIN_ACTIVE &&
                !task.react_last_step_name.empty();
        const std::string context_summary =
                use_pinned_turn_state ? std::string() : shared_context_summary_text_for_task(task);
        const std::string pinned_turn_state = react_pinned_turn_state_text(task);
        const std::string stage_name = react_tool_stage_name(task.react_tool_stage);
        const bool first_tool_call_required = react_turn_requires_first_tool_call(task);
        const bool tool_stage_select =
                task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_TOOL ||
                (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_NONE && first_tool_call_required);
        const std::string effective_stage_name = tool_stage_select ? "select_tool_family" : stage_name;
        const bool tool_stage_select_method = task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_METHOD;
        const bool tool_stage_emit = task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL;
        const bool tool_stage_decide_after_tool = task.react_tool_stage == SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL;
        const bool tool_stage_emit_response = task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE;
        const bool latest_tool_failure = latest_tool_observation_indicates_failure(task);
        const std::string expected_response_action = react_terminal_action_label(task.react_pending_terminal_action);
        const std::string emit_response_contract =
                std::string("{\"action\":\"") +
                (expected_response_action.empty() ? "answer_or_ask" : expected_response_action) +
                "\",\"assistant_text\":\"...\"}";
        const std::string selected_tool_context =
                task.react_selected_tool_family_id.empty() ?
                        std::string() :
                        "Selected tool family for this continuation: " + task.react_selected_tool_family_id + "\n";
        const std::string selected_method_context =
                task.react_selected_method_name.empty() ?
                        std::string() :
                        "Selected method for this continuation: " + task.react_selected_method_name + "\n";
        if (task.react_origin == SERVER_REACT_ORIGIN_DMN) {
            phase_system.content =
                    std::string(
                    "Use only the canonical shared cognitive context as history. "
                    "Produce one authoritative hidden DMN ReAct control step.\n"
                    "The prompt carries the accumulated staged history for the current turn in chronological order. Continue from that preserved history instead of restarting the turn.\n"
                    "The assistant reply is prefilled with the hidden reasoning block for this stage. "
                    "Treat that hidden reasoning as explanatory context only; the authoritative controller action for staged phases must be emitted in the visible JSON payload.\n") +
                    (tool_stage_select ?
                            std::string(
                    "This continuation is in staged tool protocol phase select_tool_family. "
                    "The hidden reasoning block for this stage is already prefilled. "
                    "Immediately after it, emit exactly one JSON object in the shape "
                    "{\"action\":\"select_tool\",\"tool_family_id\":\"...\"}. Do not emit method names, arguments, markdown, XML, or extra prose.\n") :
                    tool_stage_select_method ?
                            std::string(
                    "This continuation is in staged tool protocol phase select_method. "
                    "The hidden reasoning block for this stage is already prefilled. "
                    "Immediately after it, emit exactly one JSON object in the shape "
                                    "{\"action\":\"select_method\",\"method_name\":\"...\"}. Do not emit arguments, markdown, XML, or extra prose.\n") :
                    tool_stage_emit ?
                            std::string(
                                    "This continuation is in staged tool protocol phase emit_arguments. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object containing action=\"act\" plus only the arguments for the selected method. "
                                    "Do not restate the tool family or method. Do not emit markdown, XML, or extra prose.\n") :
                    tool_stage_decide_after_tool ?
                            std::string(
                                    "This continuation is in staged tool protocol phase decide_after_tool. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object in the shape "
                                    "{\"action\":\"decide\",\"decision\":\"answer|ask|select_tool|wait\"}. "
                                    "Do not emit visible prose, markdown, XML, or a tool payload during this stage.\n") :
                    tool_stage_emit_response ?
                            std::string(
                                    "This continuation is in staged tool protocol phase emit_response. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object in the shape " +
                                    emit_response_contract + ". "
                                    "Do not emit markdown, XML, bullets, or extra prose outside the JSON object.\n") :
                            std::string(
                                    "Your Action line must be exactly one of:\n"
                                    "Action: select_tool|internal_write|wait\n"
                                    "If Action is select_tool, select the appropriate tool based on the original request and current admitted context, then complete the prefilled selector line immediately after the hidden reasoning. ")) +
                    "If Action is internal_write, put the internal reflection in visible assistant content. "
                    "If Action is wait, leave visible assistant content empty. " +
                    (task.react_resuming_from_tool_result ?
                            "A completed tool observation was just admitted. Prefer act or internal_write from that observation. "
                            "Only choose wait if another already-issued external tool is still outstanding.\n" :
                            "") +
                    "Do not invent any parallel transcript or selector policy.\n\n" +
                    selected_tool_context +
                    selected_method_context +
                    (emotive_context.empty() ? std::string() : emotive_context + "\n") +
                    "Current staged tool phase: " + effective_stage_name + "\n" +
                    (pinned_turn_state.empty() ? std::string() : pinned_turn_state + "\n\n") +
                    (context_summary.empty() ? std::string() : context_summary + "\n\n") +
                    (task.react_retry_feedback.empty() ? std::string() :
                            "Previous control step failed validation: " + task.react_retry_feedback + "\n\n") +
                    (staged_guidance.empty() ? std::string("No tools are currently available.") : staged_guidance);
        } else {
            phase_system.content =
                    std::string(
                    "Use only the canonical shared cognitive context as history. "
                    "Produce one authoritative hidden active ReAct control step.\n"
                    "The prompt carries the accumulated staged history for the current turn in chronological order. Continue from that preserved history instead of restarting the turn.\n"
                    "The assistant reply is prefilled with the hidden reasoning block for this stage. "
                    "Treat that hidden reasoning as explanatory context only; the authoritative controller action for staged phases must be emitted in the visible JSON payload.\n"
                    "If Action is answer or ask, put only the user-visible reply in visible assistant content. "
                    "Any user-facing text must be plain prose only: no markdown emphasis, no code fences, no bullets, no numbered lists, and no 'as an AI' disclaimers. "
                    "If Action is wait, leave visible assistant content empty. "
                    "Only choose Action: answer or Action: ask if the needed content is already grounded in canonical shared context, "
                    "Telegram dialogue memory, or prior admitted tool observations. "
                    "For current, live, dated, external, runtime-state, repository-state, or otherwise mutable facts, strongly prefer continuing the staged tool protocol unless the exact needed answer is already present in canonical context. "
                    "Use only the structured tool families exposed for this turn; do not invent or substitute raw host-local shell execution as a tool path. "
                    "Do not disclaim lack of access when a relevant tool is available. "
                    "Do not answer from unsupported internal recall or habit memory.\n") +
                    (tool_stage_select ?
                            std::string(
                                    "This continuation is in staged tool protocol phase select_tool_family. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object in the shape "
                                    "{\"action\":\"select_tool\",\"tool_family_id\":\"...\"}. Do not answer directly, ask the user for available-tool information, or emit method or argument payloads.\n") :
                    tool_stage_select_method ?
                            std::string(
                                    "This continuation is in staged tool protocol phase select_method. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object in the shape "
                                    "{\"action\":\"select_method\",\"method_name\":\"...\"}. Do not answer directly, and do not emit argument payloads.\n") :
                    tool_stage_emit ?
                            std::string(
                                    "This continuation is in staged tool protocol phase emit_arguments. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object containing action=\"act\" plus only the arguments for the selected method. "
                                    "Do not restate the tool family or method. Do not emit markdown, XML, or extra prose.\n") :
                    tool_stage_decide_after_tool ?
                            std::string(
                                    "This continuation is in staged tool protocol phase decide_after_tool. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object in the shape "
                                    "{\"action\":\"decide\",\"decision\":\"answer|ask|select_tool|wait\"}. "
                                    "Choose select_tool if another tool invocation is still needed. "
                                    "Do not emit visible prose or a tool payload during this stage.\n") :
                    tool_stage_emit_response ?
                            std::string(
                                    "This continuation is in staged tool protocol phase emit_response. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object in the shape " +
                                    emit_response_contract + ". "
                                    "The assistant_text value must contain only the final plain-prose user-facing reply for this step. "
                                    "If the latest admitted tool observation shows a successful external action or queued external command, state that the action has already been started, queued, sent, or completed as appropriate. "
                                    "Do not describe a successfully performed tool action as something the user can do later or with some separate command. "
                                    "Do not emit markdown, bullets, numbered lists, or extra prose outside the JSON object.\n") :
                    first_tool_call_required ?
                            std::string(
                                    "This turn requires fresh tool grounding before any direct answer. "
                                    "Select the appropriate tool based on the original request and current admitted context. "
                                    "The hidden reasoning block for this stage is already prefilled. "
                                    "Immediately after it, emit exactly one JSON object in the shape "
                                    "{\"action\":\"select_tool\",\"tool_family_id\":\"...\"}. "
                                    "Do not answer directly, do not ask the user for information the available tools can obtain, and do not emit method or argument payloads during this stage.\n") :
                            std::string(
                                    "Your Action line must be exactly one of:\n"
                                    "Action: answer|ask|select_tool|wait\n"
                                    "If Action is select_tool, select the appropriate tool based on the original request and current admitted context. "
                                    "If Action is select_tool, emit exactly one JSON object with only the chosen tool family id immediately after the hidden reasoning. "
                                    "Do not emit a method selection or argument payload during this stage.\n")) +
                    (react_turn_requires_retry_tool_escalation(task) ?
                            "This mutable turn has already failed without using a tool. "
                            "On this step, you must continue the staged tool protocol and must not answer directly. "
                            "Do not answer directly and do not ask the user for information the available tools can obtain.\n" :
                            "") +
                    (task.react_resuming_from_tool_result ?
                            "A completed tool observation was just admitted. Based on that observation together with other admitted canonical context, choose answer, ask, or continue the staged tool protocol if more grounding is still needed. "
                            "Any answer or question must synthesize the observed tool results together with other admitted canonical context rather than unsupported internal recall. "
                            "If the observation is insufficient, continue the staged tool protocol instead of guessing. "
                            "If the observation shows that a requested external action already succeeded or that a service accepted and started the requested command, answer from that completed action state instead of framing it as future advice. "
                            "Do not choose wait unless another already-issued external tool is still outstanding.\n" :
                            "If the latest user turn requests current, live, dated, or otherwise external facts and a relevant tool is available, continue the staged tool protocol unless the exact needed answer is already present in canonical context.\n") +
                    (latest_tool_failure ?
                            "The latest admitted tool observation indicates a concrete tool or service failure. If another available tool step could recover or verify the request, continue the staged tool protocol. Do not claim that external tool access is unavailable when a tool was just invoked; instead summarize the observed failure or continue with another relevant tool step.\n" :
                            "") +
                    (!task.react_resuming_from_tool_result ?
                            "If the canonical context already contains the exact answer, answer directly and keep the answer grounded in that admitted context.\n" :
                            "") +
                    "Do not invent any parallel transcript or selector policy.\n\n" +
                    selected_tool_context +
                    selected_method_context +
                    (emotive_context.empty() ? std::string() : emotive_context + "\n") +
                    "Current staged tool phase: " + effective_stage_name + "\n" +
                    (pinned_turn_state.empty() ? std::string() : pinned_turn_state + "\n\n") +
                    (context_summary.empty() ? std::string() : context_summary + "\n\n") +
                    (task.react_retry_feedback.empty() ? std::string() :
                            "Previous control step failed validation: " + task.react_retry_feedback + "\n\n") +
                    (staged_guidance.empty() ? std::string("No tools are currently available.") : staged_guidance);
        }
        auto build_prompt_messages = [&](size_t history_start_index) {
            std::vector<common_chat_msg> prompt_messages =
                    canonical_react_messages(task, history_start_index);
            prompt_messages.insert(prompt_messages.begin(), phase_system);
            prompt_messages.push_back(common_chat_msg{
                    /*.role =*/ "assistant",
                    /*.content =*/ task.react_assistant_prefill,
                    /*.content_parts =*/ {},
                    /*.tool_calls =*/ {},
                    /*.reasoning_content =*/ {},
                    /*.tool_name =*/ {},
                    /*.tool_call_id =*/ {},
            });
            return prompt_messages;
        };

        auto build_chat_completion = [&](size_t history_start_index) {
            common_chat_templates_inputs inputs;
            inputs.messages = build_prompt_messages(history_start_index);
            inputs.tools = {};
            inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
            inputs.use_jinja = chat_params.use_jinja;
            inputs.parallel_tool_calls = false;
            inputs.add_generation_prompt = true;
            inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
            inputs.enable_thinking = false;
            inputs.chat_template_kwargs = chat_params.chat_template_kwargs;
            return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
        };

        const int32_t raw_prompt_budget =
                llama_n_ctx_seq(ctx) > 0 ? (int32_t) llama_n_ctx_seq(ctx) : n_ctx;
        const int32_t prompt_token_budget = std::max<int32_t>(1024, raw_prompt_budget - 1024);
        size_t history_start_index = 0;
        common_chat_params chat_completion = build_chat_completion(history_start_index);
        while (history_start_index < task.react_history.size()) {
            std::vector<server_tokens> prompt_tokens = tokenize_input_prompts(
                    vocab,
                    nullptr,
                    json(chat_completion.prompt),
                    true,
                    true);
            if (prompt_tokens.empty()) {
                return false;
            }
            size_t sizing_token_count = prompt_tokens.front().size();
            if (!core_system_prompt_prefix_tokens.empty() || !emotive_context.empty()) {
                llama_tokens prefix_tokens = core_system_prompt_prefix_tokens;
                if (!emotive_context.empty()) {
                    const std::string runtime_prefix = std::string("System:\n") + emotive_context + "\n\n";
                    const llama_tokens emotive_tokens =
                            common_tokenize(vocab, runtime_prefix, false, true);
                    prefix_tokens.insert(prefix_tokens.end(), emotive_tokens.begin(), emotive_tokens.end());
                }
                sizing_token_count += prefix_tokens.size();
            }
            if (sizing_token_count <= (size_t) prompt_token_budget) {
                break;
            }
            history_start_index += 1;
            chat_completion = build_chat_completion(history_start_index);
        }
        if (history_start_index > 0) {
            SRV_INF("trimmed %zu oldest staged ReAct history entries to fit prompt budget=%d tokens\n",
                    history_start_index,
                    prompt_token_budget);
        }
        if (!apply_chat_prompt_to_task(task, chat_completion)) {
            return false;
        }

        task.params.sampling.grammar.clear();
        task.params.sampling.grammar_lazy = false;
        task.params.sampling.grammar_triggers.clear();
        if (tool_stage_select_method) {
            std::vector<server_openclaw_tool_method_contract> methods;
            if (!openclaw_fabric.build_tool_method_contracts(
                        task.react_selected_tool_family_id,
                        &methods,
                        available_spec_indexes.empty() ? nullptr : &available_spec_indexes)) {
                return false;
            }
            std::vector<std::string> method_names;
            method_names.reserve(methods.size());
            for (const auto & method : methods) {
                method_names.push_back(method.method_name);
            }
            task.params.sampling.grammar = json_schema_to_grammar(
                    react_selector_schema(
                            "select_method",
                            "method_name",
                            "The selected method id for the already selected tool family.",
                            method_names));
        } else if (tool_stage_select) {
            std::vector<server_openclaw_tool_family_contract> families;
            if (!openclaw_fabric.build_tool_family_contracts(
                        &families,
                        available_spec_indexes.empty() ? nullptr : &available_spec_indexes)) {
                return false;
            }
            std::vector<std::string> family_ids;
            family_ids.reserve(families.size());
            for (const auto & family : families) {
                family_ids.push_back(family.tool_family_id);
            }
            task.params.sampling.grammar = json_schema_to_grammar(
                    react_selector_schema(
                            "select_tool",
                            "tool_family_id",
                            "The selected tool family id for this step.",
                            family_ids));
        } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL) {
            server_openclaw_tool_argument_contract contract;
            if (!openclaw_fabric.build_tool_argument_contract(
                        task.react_selected_tool_family_id,
                        task.react_selected_method_name,
                        &contract,
                        available_spec_indexes.empty() ? nullptr : &available_spec_indexes)) {
                return false;
            }
            task.params.sampling.grammar = json_schema_to_grammar(
                    react_wrap_method_argument_schema(contract.input_schema, "act"));
        } else if (tool_stage_decide_after_tool) {
            task.params.sampling.grammar = json_schema_to_grammar(
                    react_selector_schema(
                            "decide",
                            "decision",
                            "The next controller decision after the completed tool observation.",
                            {"answer", "ask", "select_tool", "wait"}));
        } else if (tool_stage_emit_response) {
            const std::string expected_action = react_terminal_action_label(task.react_pending_terminal_action);
            task.params.sampling.grammar = json_schema_to_grammar(json{
                    {"type", "object"},
                    {"additionalProperties", false},
                    {"required", json::array({"action", "assistant_text"})},
                    {"properties", json{
                            {"action", json{
                                    {"type", "string"},
                                    {"enum", json::array({expected_action})},
                                    {"description", "The controller action required for this staged response phase."}
                            }},
                            {"assistant_text", json{
                                    {"type", "string"},
                                    {"description", "The plain-prose user-facing response for this step."}
                            }}
                    }}
            });
        }

        task.react_iteration += 1;
        return true;
    }

    bool parse_generated_chat_message(
            const server_task & task,
            const std::string & text,
            common_chat_msg * out_msg,
            std::string * out_tool_xml = nullptr,
            std::string * out_planner_reasoning = nullptr) const {
        if (!out_msg) {
            return false;
        }
        const std::string parse_text = task.react_assistant_prefill.empty() ? text : task.react_assistant_prefill + text;
        if (!task.react_tools.empty()) {
            server_openclaw_parsed_tool_call parsed = {};
            const std::vector<int32_t> selected_spec_indexes = react_available_spec_indexes(task);
            std::string xml_error;
            if (openclaw_fabric.parse_tool_call_xml(
                        parse_text,
                        &parsed,
                        nullptr,
                        nullptr,
                        selected_spec_indexes.empty() ? nullptr : &selected_spec_indexes,
                        &xml_error)) {
                *out_msg = parsed.message;
                if (out_tool_xml) {
                    *out_tool_xml = parsed.captured_tool_xml;
                }
                if (out_planner_reasoning) {
                    *out_planner_reasoning = parsed.captured_planner_reasoning;
                }
                return !out_msg->empty();
            }

            if (openclaw_fabric.recover_tool_call_xml(
                        parse_text,
                        &parsed,
                        nullptr,
                        nullptr,
                        selected_spec_indexes.empty() ? nullptr : &selected_spec_indexes,
                        &xml_error)) {
                *out_msg = parsed.message;
                if (out_tool_xml) {
                    *out_tool_xml = parsed.captured_tool_xml;
                }
                if (out_planner_reasoning) {
                    *out_planner_reasoning = parsed.captured_planner_reasoning;
                }
                return !out_msg->empty();
            }

            const std::string sanitized = openclaw_fabric.strip_tool_call_xml_markup(parse_text);
            if (sanitized != trim_ascii_copy(parse_text)) {
                std::string reasoning_text;
                std::string visible_text;
                split_hidden_reasoning_text(vocab, sanitized, &reasoning_text, &visible_text);
                out_msg->role = "assistant";
                out_msg->content = visible_text;
                out_msg->reasoning_content = reasoning_text;
                if (out_tool_xml) {
                    out_tool_xml->clear();
                }
                if (out_planner_reasoning) {
                    *out_planner_reasoning = reasoning_text;
                }
                return !out_msg->empty();
            }
        }
        task_result_state state(task.params.chat_parser_params);
        std::vector<common_chat_msg_diff> diffs;
        *out_msg = state.update_chat_msg(parse_text, false, diffs);
        if (out_msg->tool_calls.empty() && !task.react_tools.empty()) {
            const std::string trimmed = trim_ascii_copy(parse_text);
            const auto parse_fallback_call = [&](const std::string & candidate, std::string * out_prefix) -> bool {
                for (const auto & tool : task.react_tools) {
                    const std::string prefix = tool.name + "(";
                    if (!string_starts_with(candidate, prefix) || candidate.size() <= prefix.size() + 1 || candidate.back() != ')') {
                        continue;
                    }
                    const std::string arguments = trim_ascii_copy(candidate.substr(prefix.size(), candidate.size() - prefix.size() - 1));
                    if (arguments.empty()) {
                        continue;
                    }
                    try {
                        (void) json::parse(arguments);
                    } catch (const std::exception &) {
                        continue;
                    }
                    out_msg->role = "assistant";
                    out_msg->content.clear();
                    out_msg->reasoning_content = out_prefix ? trim_ascii_copy(*out_prefix) : std::string();
                    out_msg->tool_calls = { common_chat_tool_call { tool.name, arguments, "" } };
                    return true;
                }
                return false;
            };

            std::string prefix_content;
            if (!trimmed.empty()) {
                const size_t last_break = trimmed.find_last_of('\n');
                if (last_break != std::string::npos) {
                    prefix_content = trim_ascii_copy(trimmed.substr(0, last_break));
                    const std::string last_line = trim_ascii_copy(trimmed.substr(last_break + 1));
                    (void) parse_fallback_call(last_line, &prefix_content);
                } else {
                    (void) parse_fallback_call(trimmed, nullptr);
                }
            }
        }
        std::string reasoning_text;
        std::string visible_text;
        split_hidden_reasoning_text(vocab, parse_text, &reasoning_text, &visible_text);
        if (!reasoning_text.empty()) {
            out_msg->role = "assistant";
            if (out_msg->reasoning_content.empty()) {
                out_msg->reasoning_content = reasoning_text;
            }
            if (out_msg->tool_calls.empty()) {
                out_msg->content = std::move(visible_text);
            }
        }
        if (out_tool_xml) {
            out_tool_xml->clear();
        }
        if (out_planner_reasoning) {
            *out_planner_reasoning = trim_ascii_copy(out_msg->reasoning_content);
        }
        return !out_msg->empty();
    }

    struct parsed_react_step {
        bool valid = false;
        int32_t action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
        int32_t tool_protocol_signal = 0;
        common_chat_msg assistant_msg;
        std::string tool_xml;
        std::string planner_reasoning;
        std::vector<std::string> planner_reasoning_blocks;
        std::string action_source;
        std::string selected_tool_family_id;
        std::string selected_tool_method_name;
        std::string selected_capability_id;
        std::string error;
    };

    static std::string react_limit_turn_state_text(const std::string & text, size_t max_chars = 32768) {
        const std::string trimmed = trim_ascii_copy(text);
        if (trimmed.size() <= max_chars) {
            return trimmed;
        }
        return trimmed.substr(0, max_chars) + "\n[truncated pinned turn state]";
    }

    static std::string react_json_text(const json & payload) {
        return react_limit_turn_state_text(safe_json_to_str(payload));
    }

    static void record_react_last_step(
            server_task & task,
            const std::string & name,
            const std::string & output,
            const std::string & result) {
        task.react_last_step_name = name;
        task.react_last_step_output = react_limit_turn_state_text(output);
        task.react_last_step_result = react_limit_turn_state_text(result);
    }

    static std::string react_history_entry_text(const server_react_history_entry & entry) {
        std::ostringstream oss;
        if (!trim_ascii_copy(entry.phase).empty()) {
            oss << "Phase: " << trim_ascii_copy(entry.phase) << "\n";
        }
        if (!trim_ascii_copy(entry.kind).empty()) {
            oss << "Kind: " << trim_ascii_copy(entry.kind) << "\n";
        }
        oss << trim_ascii_copy(entry.text);
        return trim_ascii_copy(oss.str());
    }

    static void append_react_history_entry(
            server_task & task,
            const std::string & phase,
            const std::string & kind,
            const std::string & text) {
        const std::string trimmed_text = react_limit_turn_state_text(text);
        if (trimmed_text.empty()) {
            return;
        }

        server_react_history_entry entry = {};
        entry.phase = trim_ascii_copy(phase);
        entry.kind = trim_ascii_copy(kind);
        entry.text = trimmed_text;
        task.react_history.push_back(std::move(entry));
    }

    std::vector<common_chat_msg> react_history_messages(
            const server_task & task,
            size_t history_start_index = 0) const {
        std::vector<common_chat_msg> messages;
        if (history_start_index >= task.react_history.size()) {
            return messages;
        }

        messages.reserve(task.react_history.size() - history_start_index);
        for (size_t i = history_start_index; i < task.react_history.size(); ++i) {
            const server_react_history_entry & entry = task.react_history[i];
            const std::string content = react_history_entry_text(entry);
            if (content.empty()) {
                continue;
            }

            common_chat_msg msg;
            msg.role = trim_ascii_copy(entry.kind) == "tool_result" ? "tool" : "assistant";
            msg.content = content;
            messages.push_back(std::move(msg));
        }
        return messages;
    }

    std::string react_pinned_turn_state_text(const server_task & task) const {
        std::ostringstream oss;
        if (!task.foreground_text.empty()) {
            oss << "Pinned original request:\n" << task.foreground_text << "\n";
        }
        if (!task.react_history.empty()) {
            oss << "Pinned staged history entries: " << task.react_history.size() << "\n";
        }
        if (!task.react_last_step_name.empty()) {
            oss << "Pinned last completed step: " << task.react_last_step_name << "\n";
        }
        if (!task.react_last_step_output.empty()) {
            oss << "Pinned last step output:\n" << task.react_last_step_output << "\n";
        }
        if (!task.react_last_step_result.empty()) {
            oss << "Pinned last step result:\n" << task.react_last_step_result << "\n";
        }
        if (!task.react_last_tool_observation.empty()) {
            oss << "Pinned latest tool observation:\n" << task.react_last_tool_observation << "\n";
        }
        return trim_ascii_copy(oss.str());
    }

    void capture_react_stage_provenance(
            const server_task & task,
            const char * source,
            const char * outcome,
            const parsed_react_step * step,
            const std::string & raw_generation,
            const std::string & error = std::string()) {
        vicuna_progress_snapshot progress = {};
        llama_self_model_state_info model_state = {};
        llama_functional_lora_trace functional_trace = {};
        llama_process_functional_trace process_trace = {};
        if (!collect_progress_snapshot(&progress, &model_state, &functional_trace, &process_trace)) {
            return;
        }

        const char * phase_name = react_tool_stage_name(task.react_tool_stage);
        if (step) {
            if (step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_TOOL) {
                phase_name = "select_tool_family";
            } else if (step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_METHOD) {
                phase_name = "select_method";
            } else if (step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_EMIT_TOOL_CALL) {
                phase_name = "emit_arguments";
            } else if (step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_DECIDE_AFTER_TOOL) {
                phase_name = "decide_after_tool";
            } else if (step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_EMIT_RESPONSE) {
                phase_name = "emit_response";
            }
        }
        json stage = {
            {"origin", task.react_origin},
            {"phase", phase_name},
            {"retry_count", task.react_retry_count},
            {"stage_retry_count", task.react_stage_retry_count},
            {"resuming_from_tool_result", task.react_resuming_from_tool_result},
            {"selected_tool_family_id", task.react_selected_tool_family_id},
            {"selected_method_name", task.react_selected_method_name},
            {"selected_capability_id", task.react_selected_capability_id},
            {"pending_terminal_action", task.react_pending_terminal_action},
            {"last_step_name", task.react_last_step_name},
            {"last_tool_family_id", task.react_last_tool_family_id},
            {"last_tool_method_name", task.react_last_tool_method_name},
            {"last_tool_capability_id", task.react_last_tool_capability_id},
        };
        json payload = {
            {"stage", stage},
            {"outcome", outcome ? outcome : ""},
            {"raw_generation", raw_generation},
            {"pinned_turn_state", {
                    {"original_request", task.foreground_text},
                    {"history_size", task.react_history.size()},
                    {"last_step_name", task.react_last_step_name},
                    {"last_step_output", task.react_last_step_output},
                    {"last_step_result", task.react_last_step_result},
                    {"last_tool_family_id", task.react_last_tool_family_id},
                    {"last_tool_method_name", task.react_last_tool_method_name},
                    {"last_tool_capability_id", task.react_last_tool_capability_id},
                    {"last_tool_observation", task.react_last_tool_observation},
            }},
            {"self_model", self_model_summary_to_json(model_state)},
            {"functional", functional_trace_summary_to_json(functional_trace)},
            {"process_functional", process_trace_summary_to_json(process_trace)},
        };
        if (task.react_origin == SERVER_REACT_ORIGIN_ACTIVE) {
            payload["active_loop"] = active_trace_summary_to_json(task.active_trace);
        } else if (task.react_origin == SERVER_REACT_ORIGIN_DMN) {
            payload["dmn"] = dmn_trace_summary_to_json(task.dmn_trace);
        }
        if (step) {
            payload["step"] = {
                {"action", step->action},
                {"action_source", step->action_source},
                {"tool_protocol_signal", step->tool_protocol_signal},
                {"reasoning_text", step->planner_reasoning},
                {"visible_text", step->assistant_msg.content},
                {"stage_payload", step->tool_xml},
                {"selected_tool_family_id", step->selected_tool_family_id},
                {"selected_method_name", step->selected_tool_method_name},
                {"selected_capability_id", step->selected_capability_id},
                {"pending_terminal_action", task.react_pending_terminal_action},
            };
        }
        if (!error.empty()) {
            payload["error"] = error;
        }
        if (!task.react_last_failure_class.empty()) {
            payload["last_failure_class"] = task.react_last_failure_class;
        }
        if (!task.react_last_failure_detail.empty()) {
            payload["last_failure_detail"] = task.react_last_failure_detail;
        }
        (void) append_provenance_event("react_stage", source ? source : "react", payload, progress);
    }

    enum react_tool_protocol_signal {
        REACT_TOOL_PROTOCOL_SIGNAL_NONE = 0,
        REACT_TOOL_PROTOCOL_SIGNAL_SELECT_TOOL = 1,
        REACT_TOOL_PROTOCOL_SIGNAL_SELECT_METHOD = 2,
        REACT_TOOL_PROTOCOL_SIGNAL_EMIT_TOOL_CALL = 3,
        REACT_TOOL_PROTOCOL_SIGNAL_DECIDE_AFTER_TOOL = 4,
        REACT_TOOL_PROTOCOL_SIGNAL_EMIT_RESPONSE = 5,
    };

    std::string validate_authoritative_react_terminal_policy(
            const server_task & task,
            const parsed_react_step & step) const {
        if (task.react_origin != SERVER_REACT_ORIGIN_ACTIVE) {
            return {};
        }

        if (step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER ||
                step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK) {
            const std::string visible = trim_ascii_copy(step.assistant_msg.content);
            if (authoritative_reply_is_procedural_non_answer(visible)) {
                return "visible reply described intended work or lack of access instead of completing the turn";
            }
            if (user_facing_text_violates_plain_prose_policy(visible)) {
                return "visible reply violated the plain-prose policy for user-facing text";
            }
        }

        if (step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER &&
                react_turn_requires_first_tool_call(task)) {
            return "latest active user turn requires fresh tool grounding before any direct answer";
        }
        if ((step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK ||
                step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT) &&
                react_turn_requires_first_tool_call(task)) {
            return "latest active user turn requires staged tool selection before asking or waiting";
        }

        if (step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ACT &&
                task.react_resuming_from_tool_result &&
                canonical_context_has_direct_tool_answer_observation(task)) {
            return "latest admitted tool observation already contains a direct answer; synthesize it before issuing another tool call";
        }

        if ((step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER ||
                step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK) &&
                task.react_resuming_from_tool_result &&
                foreground_request_requires_external_action(task.foreground_text) &&
                latest_tool_step_was_read_only(task)) {
            return "latest completed tool step was read-only, but the original request still requires an external action; continue the staged tool protocol";
        }

        return {};
    }

    static std::string react_normalize_label(std::string label) {
        label = trim_ascii_copy(label);
        std::transform(label.begin(), label.end(), label.begin(), [](unsigned char ch) {
            if (ch == '-' || ch == ' ') {
                return '_';
            }
            return (char) std::tolower(ch);
        });
        return label;
    }

    static std::string react_extract_action_label(const std::string & reasoning) {
        std::istringstream in(reasoning);
        std::string line;
        while (std::getline(in, line)) {
            std::string trimmed = trim_ascii_copy(line);
            if (trimmed.empty()) {
                continue;
            }
            std::string lowered = react_normalize_label(trimmed);
            if (!string_starts_with(lowered, "action:")) {
                continue;
            }
            return react_normalize_label(trimmed.substr(trimmed.find(':') + 1));
        }
        return {};
    }

    static int32_t react_parse_terminal_action_label(const std::string & action) {
        if (action == "answer") {
            return LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER;
        }
        if (action == "ask") {
            return LLAMA_AUTHORITATIVE_REACT_ACTION_ASK;
        }
        if (action == "act") {
            return LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
        }
        if (action == "wait") {
            return LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT;
        }
        if (action == "internal_write" || action == "internalwrite" || action == "reflect") {
            return LLAMA_AUTHORITATIVE_REACT_ACTION_INTERNAL_WRITE;
        }
        return LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
    }

    static bool react_reasoning_has_explanatory_text(const std::string & reasoning) {
        std::istringstream in(reasoning);
        std::string line;
        while (std::getline(in, line)) {
            std::string trimmed = trim_ascii_copy(line);
            if (trimmed.empty()) {
                continue;
            }
            std::string lowered = react_normalize_label(trimmed);
            if (string_starts_with(lowered, "action:")) {
                continue;
            }
            if (string_starts_with(lowered, "thought:")) {
                trimmed = trim_ascii_copy(trimmed.substr(trimmed.find(':') + 1));
            }
            for (char ch : trimmed) {
                if (std::isalpha((unsigned char) ch)) {
                    return true;
                }
            }
        }
        return false;
    }

    static std::string react_capability_tool_family_id(const server_openclaw_capability_runtime & capability) {
        return !capability.descriptor.tool_family_id.empty() ?
                capability.descriptor.tool_family_id :
                capability.descriptor.tool_name;
    }

    static std::string react_capability_method_name(const server_openclaw_capability_runtime & capability) {
        return !capability.descriptor.method_name.empty() ?
                capability.descriptor.method_name :
                capability.descriptor.tool_name;
    }

    void populate_recovered_tool_metadata(parsed_react_step * out_step) const {
        if (!out_step || out_step->assistant_msg.tool_calls.empty()) {
            return;
        }

        const common_chat_tool_call & tool_call = out_step->assistant_msg.tool_calls.front();
        const std::string emitted_tool_name = trim_ascii_copy(tool_call.name);
        const server_openclaw_capability_runtime * capability =
                emitted_tool_name.empty() ? nullptr : openclaw_fabric.capability_by_tool_name(emitted_tool_name);
        if (!capability) {
            return;
        }

        out_step->selected_tool_family_id = react_capability_tool_family_id(*capability);
        out_step->selected_tool_method_name = react_capability_method_name(*capability);
        out_step->selected_capability_id = capability->descriptor.capability_id;
    }

    bool infer_authoritative_react_step_without_action_label(
            const server_task & task,
            const std::string & text,
            parsed_react_step * out_step) const {
        if (!out_step) {
            return false;
        }

        common_chat_msg fallback_msg = out_step->assistant_msg;
        std::string fallback_reasoning = out_step->planner_reasoning;
        if (!task.react_tools.empty()) {
            common_chat_msg parsed_msg = {};
            std::string parsed_tool_xml;
            std::string parsed_planner_reasoning;
            if (parse_generated_chat_message(
                        task,
                        text,
                        &parsed_msg,
                        &parsed_tool_xml,
                        &parsed_planner_reasoning)) {
                if (!parsed_msg.tool_calls.empty()) {
                    out_step->action = LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
                    out_step->tool_protocol_signal = REACT_TOOL_PROTOCOL_SIGNAL_EMIT_TOOL_CALL;
                    out_step->assistant_msg = std::move(parsed_msg);
                    out_step->tool_xml = trim_ascii_copy(parsed_tool_xml);
                    out_step->planner_reasoning =
                            trim_ascii_copy(parsed_planner_reasoning.empty() ?
                                                    out_step->assistant_msg.reasoning_content :
                                                    parsed_planner_reasoning);
                    out_step->action_source = "recovered_tool_call";
                    populate_recovered_tool_metadata(out_step);
                    return true;
                }
                fallback_msg = std::move(parsed_msg);
                fallback_reasoning =
                        trim_ascii_copy(parsed_planner_reasoning.empty() ?
                                                fallback_msg.reasoning_content :
                                                parsed_planner_reasoning);
            }
        }

        const int32_t inferred_action = infer_authoritative_action_from_visible_surface(
                task.react_origin == SERVER_REACT_ORIGIN_DMN,
                fallback_msg.content,
                task.react_resuming_from_tool_result,
                task.foreground_role);
        if (inferred_action == LLAMA_AUTHORITATIVE_REACT_ACTION_NONE) {
            return false;
        }

        out_step->action = inferred_action;
        out_step->assistant_msg.content = trim_ascii_copy(fallback_msg.content);
        out_step->assistant_msg.reasoning_content = trim_ascii_copy(
                fallback_msg.reasoning_content.empty() ?
                        out_step->assistant_msg.reasoning_content :
                        fallback_msg.reasoning_content);
        out_step->planner_reasoning = trim_ascii_copy(
                fallback_reasoning.empty() ?
                        out_step->assistant_msg.reasoning_content :
                        fallback_reasoning);
        out_step->tool_xml.clear();
        out_step->action_source =
                inferred_action == LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT ?
                        "permitted_empty_wait" :
                        "visible_terminal_text";
        return true;
    }

    bool parse_authoritative_react_step(
            const server_task & task,
            const std::string & text,
            parsed_react_step * out_step) const {
        if (!out_step) {
            return false;
        }
        *out_step = {};
        const std::string parse_text =
                task.react_assistant_prefill.empty() ? text : task.react_assistant_prefill + text;
        std::string raw_reasoning;
        std::string raw_visible;
        split_hidden_reasoning_text(vocab, parse_text, &raw_reasoning, &raw_visible, &out_step->planner_reasoning_blocks);
        out_step->assistant_msg = {};
        out_step->assistant_msg.role = "assistant";
        out_step->assistant_msg.content = trim_ascii_copy(raw_visible);
        out_step->assistant_msg.reasoning_content = trim_ascii_copy(raw_reasoning);
        out_step->planner_reasoning = trim_ascii_copy(out_step->assistant_msg.reasoning_content);
        if (out_step->planner_reasoning.empty() || !react_reasoning_has_explanatory_text(out_step->planner_reasoning)) {
            out_step->error = "authoritative ReAct control requires explanatory hidden reasoning";
            return false;
        }

        if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL) {
            int32_t decided_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
            if (!react_parse_post_tool_decision_json(
                        raw_visible,
                        &decided_action,
                        &out_step->error)) {
                return false;
            }
            out_step->action = decided_action;
            out_step->tool_protocol_signal = REACT_TOOL_PROTOCOL_SIGNAL_DECIDE_AFTER_TOOL;
            out_step->tool_xml = trim_ascii_copy(raw_visible);
            out_step->planner_reasoning = trim_ascii_copy(out_step->assistant_msg.reasoning_content);
            out_step->action_source = "visible_stage_contract";
        } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE) {
            const std::string expected_action = react_terminal_action_label(task.react_pending_terminal_action);
            if (expected_action.empty()) {
                out_step->error = "response continuation requires a pending terminal action";
                return false;
            }
            std::string assistant_text;
            if (!react_parse_terminal_response_json(
                        raw_visible,
                        expected_action,
                        &assistant_text,
                        &out_step->error)) {
                return false;
            }
            out_step->action = task.react_pending_terminal_action;
            out_step->tool_protocol_signal = REACT_TOOL_PROTOCOL_SIGNAL_EMIT_RESPONSE;
            out_step->assistant_msg.content = std::move(assistant_text);
            out_step->tool_xml = trim_ascii_copy(raw_visible);
            out_step->planner_reasoning = trim_ascii_copy(out_step->assistant_msg.reasoning_content);
            out_step->action_source = "visible_stage_contract";
        } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_METHOD) {
            server_openclaw_parsed_tool_method_selection parsed = {};
            const std::vector<int32_t> selected_spec_indexes = react_available_spec_indexes(task);
            if (!openclaw_fabric.parse_tool_method_selection_json(
                        parse_text,
                        task.react_selected_tool_family_id,
                        &parsed,
                        selected_spec_indexes.empty() ? nullptr : &selected_spec_indexes,
                        &out_step->error)) {
                return false;
            }
            if (!trim_ascii_copy(parsed.visible_prefix).empty()) {
                out_step->error = "tool-method selection must not emit visible prose";
                return false;
            }
            out_step->action = LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
            out_step->tool_protocol_signal = REACT_TOOL_PROTOCOL_SIGNAL_SELECT_METHOD;
            out_step->assistant_msg = parsed.message;
            out_step->tool_xml = parsed.xml_block;
            out_step->planner_reasoning = trim_ascii_copy(parsed.captured_planner_reasoning);
            out_step->action_source = "visible_stage_contract";
            out_step->selected_tool_family_id = parsed.tool_family_id;
            out_step->selected_tool_method_name = parsed.method_name;
            out_step->selected_capability_id = parsed.capability_id;
        } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL) {
            server_openclaw_parsed_tool_arguments parsed = {};
            const std::vector<int32_t> selected_spec_indexes = react_available_spec_indexes(task);
            if (!openclaw_fabric.parse_tool_arguments_json(
                        parse_text,
                        task.react_selected_tool_family_id,
                        task.react_selected_method_name,
                        &parsed,
                        selected_spec_indexes.empty() ? nullptr : &selected_spec_indexes,
                        &out_step->error)) {
                return false;
            }
            out_step->action = LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
            out_step->tool_protocol_signal = REACT_TOOL_PROTOCOL_SIGNAL_EMIT_TOOL_CALL;
            out_step->assistant_msg = parsed.message;
            out_step->tool_xml = parsed.captured_arguments_json;
            out_step->planner_reasoning = trim_ascii_copy(parsed.captured_planner_reasoning);
            out_step->action_source = "visible_stage_contract";
            out_step->selected_tool_family_id = task.react_selected_tool_family_id;
            out_step->selected_tool_method_name = task.react_selected_method_name;
            out_step->selected_capability_id = task.react_selected_capability_id;
        } else if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_TOOL ||
                (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_NONE && react_turn_requires_first_tool_call(task))) {
            server_openclaw_parsed_tool_selection parsed = {};
            const std::vector<int32_t> selected_spec_indexes = react_available_spec_indexes(task);
            if (!openclaw_fabric.parse_tool_selection_json(
                        parse_text,
                        &parsed,
                        selected_spec_indexes.empty() ? nullptr : &selected_spec_indexes,
                        &out_step->error)) {
                return false;
            }
            if (!trim_ascii_copy(parsed.visible_prefix).empty()) {
                out_step->error = "tool selection must not emit visible prose";
                return false;
            }
            out_step->action = LLAMA_AUTHORITATIVE_REACT_ACTION_ACT;
            out_step->tool_protocol_signal = REACT_TOOL_PROTOCOL_SIGNAL_SELECT_TOOL;
            out_step->assistant_msg = parsed.message;
            out_step->tool_xml = parsed.xml_block;
            out_step->planner_reasoning = trim_ascii_copy(parsed.captured_planner_reasoning);
            out_step->action_source = "visible_stage_contract";
            out_step->selected_tool_family_id = parsed.tool_family_id;
        } else {
            const std::string action_label = react_extract_action_label(out_step->planner_reasoning);
            if (action_label.empty()) {
                if (!infer_authoritative_react_step_without_action_label(task, text, out_step)) {
                    out_step->error =
                            "hidden reasoning did not include an explicit Action label and no recoverable visible control surface was present";
                    return false;
                }
            } else {
                out_step->action = react_parse_terminal_action_label(action_label);
                if (out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_NONE) {
                    if (!infer_authoritative_react_step_without_action_label(task, text, out_step)) {
                        out_step->error = "unsupported Action label for the current authoritative ReAct stage";
                        return false;
                    }
                    out_step->action_source = "malformed_action_label_fallback";
                } else {
                    out_step->action_source = "hidden_action_label";
                }
            }
        }

        if (task.react_origin == SERVER_REACT_ORIGIN_DMN) {
            if (out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER ||
                    out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK) {
                out_step->error = "DMN may only choose act, internal_write, or wait";
                return false;
            }
            if (out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_INTERNAL_WRITE &&
                    trim_ascii_copy(out_step->assistant_msg.content).empty()) {
                out_step->error = "DMN internal_write requires visible assistant content for the internal write";
                return false;
            }
        } else {
            if (out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_INTERNAL_WRITE) {
                out_step->error = "active turns may not choose internal_write";
                return false;
            }
            if ((out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER ||
                        out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK) &&
                    out_step->tool_protocol_signal != REACT_TOOL_PROTOCOL_SIGNAL_DECIDE_AFTER_TOOL &&
                    trim_ascii_copy(out_step->assistant_msg.content).empty()) {
                out_step->error = "answer and ask actions require visible assistant content";
                return false;
            }
            if (out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT &&
                    !task.react_resuming_from_tool_result &&
                    task.foreground_role != LLAMA_SELF_STATE_EVENT_TOOL) {
                out_step->error = "active wait is only valid while integrating an earlier tool result";
                return false;
            }
        }

        const size_t tool_call_count = out_step->assistant_msg.tool_calls.size();
        const bool have_tool_call = tool_call_count > 0;
        if (out_step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_EMIT_TOOL_CALL) {
            if (tool_call_count != 1) {
                out_step->error = "final tool-call emission requires exactly one tool call";
                return false;
            }
        } else if (out_step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_TOOL ||
                out_step->tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_METHOD) {
            if (have_tool_call) {
                out_step->error = "intermediate staged tool steps must not emit final tool calls";
                return false;
            }
        } else {
            if (out_step->action == LLAMA_AUTHORITATIVE_REACT_ACTION_ACT) {
                out_step->error = "direct Action: act is not allowed; continue the staged tool protocol instead";
                return false;
            }
            if (have_tool_call) {
                out_step->error = "non-act actions must not emit tool calls";
                return false;
            }
        }

        out_step->valid = true;
        return true;
    }

    bool enqueue_dmn_react_task(const llama_dmn_tick_trace & trace) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = queue_tasks.get_new_id();
        task.params.stream = false;
        task.params.cache_prompt = false;
        task.params.n_predict = params_base.n_predict;
        task.params.sampling = params_base.sampling;
        task.params.speculative = params_base.speculative;
        task.foreground_role = LLAMA_SELF_STATE_EVENT_SYSTEM;
        task.react_origin = SERVER_REACT_ORIGIN_DMN;
        task.has_dmn_trace = true;
        task.dmn_trace = trace;

        if (!prepare_react_prompt(task)) {
            return false;
        }

        queue_tasks.post(std::move(task), true);
        return true;
    }

    std::string react_intent_hint(const server_task & /* task */) const {
        if (ctx) {
            for (int32_t i = llama_shared_cognitive_context_count(ctx) - 1; i >= 0; --i) {
                llama_shared_cognitive_context_item item = {};
                if (llama_shared_cognitive_context_get_item(ctx, i, &item) != 0 ||
                        item.kind != LLAMA_SHARED_CONTEXT_KIND_USER_MESSAGE ||
                        item.role != LLAMA_SELF_STATE_EVENT_USER) {
                    continue;
                }
                std::string text;
                if (shared_context_item_text(i, &text) && !text.empty()) {
                    return text;
                }
            }
        }
        const std::string emotive = current_emotive_moment_text();
        if (!emotive.empty()) {
            return emotive;
        }
        return shared_context_summary_text(4);
    }

    bool configure_tool_request_from_chat_call(
            server_task & task,
            const common_chat_msg & assistant_msg,
            int32_t * out_command_id,
            std::string * out_error) {
        if (out_command_id) {
            *out_command_id = -1;
        }

        const common_chat_tool_call * tool_call =
                assistant_msg.tool_calls.empty() ? nullptr : &assistant_msg.tool_calls.front();
        std::string emitted_tool_name = tool_call ? trim_ascii_copy(tool_call->name) : std::string();
        const server_openclaw_capability_runtime * capability =
                emitted_tool_name.empty() ? nullptr : openclaw_fabric.capability_by_tool_name(emitted_tool_name);
        if (!capability) {
            if (out_error) {
                *out_error = emitted_tool_name.empty() ?
                        "tool-call XML did not include a tool name" :
                        "planner selected an unknown tool: " + emitted_tool_name;
            }
            return false;
        }
        if (!trim_ascii_copy(task.react_selected_capability_id).empty() &&
                capability->descriptor.capability_id != trim_ascii_copy(task.react_selected_capability_id)) {
            if (out_error) {
                *out_error = "final staged tool call resolved to a different capability than the selected method";
            }
            return false;
        }

        const int32_t selected_spec_index = (int32_t) (capability - openclaw_fabric.capabilities().data());
        if (!react_spec_index_exposed(task, selected_spec_index)) {
            if (out_error) {
                *out_error = "planner selected a tool that was not exposed for this turn";
            }
            return false;
        }

        json arguments = json::object();
        if (tool_call && !tool_call->arguments.empty()) {
            try {
                arguments = json::parse(tool_call->arguments);
            } catch (const std::exception &) {
                arguments = json::object();
            }
        }
        if (!capability->descriptor.fixed_arguments_json.empty()) {
            json fixed_arguments = json::object();
            try {
                fixed_arguments = json::parse(capability->descriptor.fixed_arguments_json);
            } catch (const std::exception & err) {
                if (out_error) {
                    *out_error = std::string("failed to parse fixed runtime arguments for capability: ") + err.what();
                }
                return false;
            }
            if (!fixed_arguments.is_object()) {
                if (out_error) {
                    *out_error = "fixed runtime arguments for capability must be a JSON object";
                }
                return false;
            }
            for (auto it = fixed_arguments.begin(); it != fixed_arguments.end(); ++it) {
                const auto existing = arguments.find(it.key());
                if (existing != arguments.end() && *existing != it.value()) {
                    if (out_error) {
                        *out_error = "tool-call argument conflicts with fixed runtime argument: " + it.key();
                    }
                    return false;
                }
                arguments[it.key()] = it.value();
            }
        }

        int32_t command_id = -1;
        int32_t tool_job_id = -1;
        const int32_t origin =
                task.react_origin == SERVER_REACT_ORIGIN_DMN ?
                        LLAMA_COG_COMMAND_ORIGIN_DMN :
                        LLAMA_COG_COMMAND_ORIGIN_ACTIVE;
        if (task.react_origin == SERVER_REACT_ORIGIN_DMN) {
            llama_cognitive_dmn_runner_status runner = {};
            if (llama_cognitive_dmn_runner_get(ctx, &runner) == 0 && runner.pending_command_id > 0) {
                command_id = runner.pending_command_id;
            } else if (llama_cognitive_dmn_authoritative_begin_tool(
                               ctx,
                               task.dmn_trace.tick_id,
                               0,
                               std::max(task.dmn_trace.winner_score, 0.50f),
                               &command_id,
                               &tool_job_id) != 0) {
                if (out_error) {
                    *out_error = "failed to begin authoritative DMN tool command";
                }
                return false;
            }
        } else {
            llama_cognitive_active_runner_status runner = {};
            if (llama_cognitive_active_runner_get(ctx, &runner) == 0 && runner.pending_command_id > 0) {
                command_id = runner.pending_command_id;
            } else if (llama_cognitive_active_authoritative_begin_tool(
                               ctx,
                               task.active_trace.episode_id,
                               task.active_trace.reason_mask,
                               std::max(task.active_trace.winner_score, 0.50f),
                               &command_id,
                               &tool_job_id) != 0) {
                if (out_error) {
                    *out_error = "failed to begin authoritative active tool command";
                }
                return false;
            }
        }
        if (command_id <= 0 || llama_cognitive_command_rebind_tool(ctx, command_id, selected_spec_index) != 0) {
            if (out_error) {
                *out_error = "failed to bind the selected tool to the pending command";
            }
            return false;
        }

        if (tool_job_id <= 0) {
            llama_cognitive_command command = {};
            for (int32_t i = 0, n = llama_cognitive_command_count(ctx); i < n; ++i) {
                if (llama_cognitive_command_get(ctx, i, &command) == 0 && command.command_id == command_id) {
                    tool_job_id = command.tool_job_id;
                    break;
                }
            }
        }
        if (out_command_id) {
            *out_command_id = command_id;
        }
        const std::string intent_hint = react_intent_hint(task);

        if (capability->backend == SERVER_OPENCLAW_DISPATCH_LEGACY_BASH) {
            llama_bash_tool_request request =
                    make_bash_tool_request_from_config(command_id, origin, tool_job_id, intent_hint);

            std::string command_text = request.command_text;
            if (capability->descriptor.capability_id == "openclaw.tavily.web_search") {
                std::string query = trim_ascii_copy(json_value(arguments, "query", std::string()));
                if (query.empty()) {
                    query = trim_ascii_copy(intent_hint);
                }
                command_text =
                        resolve_openclaw_wrapper_command(request, "tavily-web-search") +
                        " --query-url=" + percent_encode_query(query);
                request.login_shell = false;
                std::snprintf(request.allowed_commands, sizeof(request.allowed_commands), "%s", "tavily-web-search");
                const std::string topic = trim_ascii_copy(json_value(arguments, "topic", std::string()));
                const std::string search_depth = trim_ascii_copy(json_value(arguments, "search_depth", std::string()));
                const int32_t max_results = json_value(arguments, "max_results", 0);
                const std::string time_range = trim_ascii_copy(json_value(arguments, "time_range", std::string()));
                const std::string country = trim_ascii_copy(json_value(arguments, "country", std::string()));
                const auto include_domains = json_value(arguments, "include_domains", json::array());
                const auto exclude_domains = json_value(arguments, "exclude_domains", json::array());
                if (!topic.empty()) {
                    command_text += " --topic=" + topic;
                }
                if (!search_depth.empty()) {
                    command_text += " --search-depth=" + search_depth;
                }
                if (max_results > 0) {
                    command_text += " --max-results=" + std::to_string(max_results);
                }
                if (!time_range.empty()) {
                    command_text += " --time-range=" + time_range;
                }
                if (!country.empty()) {
                    command_text += " --country=" + percent_encode_query(country);
                }
                if (include_domains.is_array() && !include_domains.empty()) {
                    std::vector<std::string> domains;
                    for (const auto & entry : include_domains) {
                        const std::string domain = trim_ascii_copy(entry.get<std::string>());
                        if (!domain.empty()) {
                            domains.push_back(domain);
                        }
                    }
                    if (!domains.empty()) {
                        std::ostringstream joined;
                        for (size_t i = 0; i < domains.size(); ++i) {
                            if (i > 0) {
                                joined << ',';
                            }
                            joined << domains[i];
                        }
                        command_text += " --include-domains=" + percent_encode_query(joined.str());
                    }
                }
                if (exclude_domains.is_array() && !exclude_domains.empty()) {
                    std::vector<std::string> domains;
                    for (const auto & entry : exclude_domains) {
                        const std::string domain = trim_ascii_copy(entry.get<std::string>());
                        if (!domain.empty()) {
                            domains.push_back(domain);
                        }
                    }
                    if (!domains.empty()) {
                        std::ostringstream joined;
                        for (size_t i = 0; i < domains.size(); ++i) {
                            if (i > 0) {
                                joined << ',';
                            }
                            joined << domains[i];
                        }
                        command_text += " --exclude-domains=" + percent_encode_query(joined.str());
                    }
                }
            } else if (string_starts_with(capability->descriptor.capability_id, "openclaw.servarr.radarr")) {
                command_text =
                        resolve_openclaw_wrapper_command(request, "radarr-api") +
                        " --payload-base64=" +
                        base64::encode(arguments.dump());
                request.login_shell = false;
                std::snprintf(request.allowed_commands, sizeof(request.allowed_commands), "%s", "radarr-api");
            } else if (string_starts_with(capability->descriptor.capability_id, "openclaw.servarr.sonarr")) {
                command_text =
                        resolve_openclaw_wrapper_command(request, "sonarr-api") +
                        " --payload-base64=" +
                        base64::encode(arguments.dump());
                request.login_shell = false;
                std::snprintf(request.allowed_commands, sizeof(request.allowed_commands), "%s", "sonarr-api");
            } else if (string_starts_with(capability->descriptor.capability_id, "openclaw.servarr.chaptarr")) {
                command_text =
                        resolve_openclaw_wrapper_command(request, "chaptarr-api") +
                        " --payload-base64=" +
                        base64::encode(arguments.dump());
                request.login_shell = false;
                std::snprintf(request.allowed_commands, sizeof(request.allowed_commands), "%s", "chaptarr-api");
            }

            command_text = trim_ascii_copy(command_text);
            if (command_text.empty()) {
                if (out_error) {
                    *out_error = "planner step did not yield a runnable bash command";
                }
                return false;
            }

            request.command_ready = true;
            std::snprintf(request.command_text, sizeof(request.command_text), "%s", command_text.c_str());
            if (!assistant_msg.content.empty()) {
                std::snprintf(request.intent_text, sizeof(request.intent_text), "%s", assistant_msg.content.c_str());
            }
            if (llama_cognitive_bash_tool_set_request(ctx, &request) != 0) {
                if (out_error) {
                    *out_error = "failed to install updated bash tool request";
                }
                return false;
            }
        } else if (capability->backend == SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY) {
            llama_cognitive_hard_memory_request request = {};
            if (llama_cognitive_hard_memory_get_request(ctx, command_id, &request) != 0) {
                request.command_id = command_id;
                request.origin = origin;
                request.tool_job_id = tool_job_id;
            }
            if (capability->tool_spec.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                const json memories = arguments.value("memories", json::array());
                if (!memories.is_array() || memories.empty()) {
                    if (out_error) {
                        *out_error = "hard-memory write tool call did not include any memories";
                    }
                    return false;
                }
                request.operation = LLAMA_COG_HARD_MEMORY_OPERATION_WRITE;
                request.write_count = 0;
                std::memset(request.container_tag, 0, sizeof(request.container_tag));
                const std::string container_tag = trim_ascii_copy(json_value(arguments, "containerTag", std::string()));
                if (!container_tag.empty()) {
                    std::snprintf(request.container_tag, sizeof(request.container_tag), "%s", container_tag.c_str());
                }
                for (const auto & item_json : memories) {
                    if (request.write_count >= LLAMA_HARD_MEMORY_MAX_PRIMITIVES) {
                        break;
                    }
                    if (!populate_hard_memory_write_item_from_json(
                                item_json,
                                &request.write_items[request.write_count])) {
                        continue;
                    }
                    ++request.write_count;
                }
                if (request.write_count <= 0) {
                    if (out_error) {
                        *out_error = "hard-memory write tool call did not include any valid memory items";
                    }
                    return false;
                }
            } else {
                request.operation = LLAMA_COG_HARD_MEMORY_OPERATION_QUERY;
                const std::string query = trim_ascii_copy(json_value(arguments, "query", std::string()));
                if (query.empty()) {
                    if (out_error) {
                        *out_error = "hard-memory tool call did not include a query";
                    }
                    return false;
                }
                std::snprintf(request.query.query, sizeof(request.query.query), "%s", query.c_str());
            }
            if (llama_cognitive_hard_memory_set_request(ctx, &request) != 0) {
                if (out_error) {
                    *out_error = "failed to install updated hard-memory request";
                }
                return false;
            }
        } else if (capability->backend == SERVER_OPENCLAW_DISPATCH_LEGACY_CODEX) {
            llama_codex_tool_request request = {};
            if (llama_cognitive_codex_tool_get_request(ctx, command_id, &request) != 0) {
                request.command_id = command_id;
                request.origin = origin;
                request.tool_job_id = tool_job_id;
                request.timeout_ms = configured_codex_tool.timeout_ms;
                request.max_stdout_bytes = configured_codex_tool.max_stdout_bytes;
                request.max_stderr_bytes = configured_codex_tool.max_stderr_bytes;
                request.dangerous_no_approval = configured_codex_tool.dangerous_no_approval;
                request.rebuild_after_changes = configured_codex_tool.rebuild_after_changes;
                request.verify_tool_access_after_rebuild = configured_codex_tool.verify_tool_access_after_rebuild;
                std::snprintf(request.codex_path, sizeof(request.codex_path), "%s", configured_codex_tool.codex_path);
                std::snprintf(request.working_directory, sizeof(request.working_directory), "%s", configured_codex_tool.working_directory);
                std::snprintf(request.rebuild_script_path, sizeof(request.rebuild_script_path), "%s", configured_codex_tool.rebuild_script_path);
                std::snprintf(request.rebuild_helper_path, sizeof(request.rebuild_helper_path), "%s", configured_codex_tool.rebuild_helper_path);
                std::snprintf(request.completion_message_path, sizeof(request.completion_message_path), "%s", configured_codex_tool.completion_message_path);
                std::snprintf(request.intent_text, sizeof(request.intent_text), "%s", intent_hint.c_str());
            }
            std::string task_text = trim_ascii_copy(json_value(arguments, "task", std::string()));
            if (task_text.empty()) {
                task_text = trim_ascii_copy(intent_hint);
            }
            if (task_text.empty()) {
                if (out_error) {
                    *out_error = "codex tool call did not include a task";
                }
                return false;
            }
            std::snprintf(request.intent_text, sizeof(request.intent_text), "%s", task_text.c_str());
            const std::string full_prompt =
                    task_text +
                    "\n\nApply the requested repository change directly in this checkout. Run the commands and tests you need. "
                    "When you finish, end with a brief plain-text summary that includes a line beginning with "
                    "'Manual requirements:' followed by either 'none' or the secrets / API keys the user must still add manually.";
            std::snprintf(request.task_prompt, sizeof(request.task_prompt), "%s", full_prompt.c_str());
            request.command_ready = true;
            if (llama_cognitive_codex_tool_set_request(ctx, &request) != 0) {
                if (out_error) {
                    *out_error = "failed to install updated codex tool request";
                }
                return false;
            }
        } else if (capability->backend == SERVER_OPENCLAW_DISPATCH_LEGACY_TELEGRAM) {
            if (capability->tool_spec.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_ASK_OPTIONS) {
                llama_telegram_ask_options_request request = {};
                if (llama_cognitive_telegram_ask_options_get_request(ctx, command_id, &request) != 0) {
                    request.command_id = command_id;
                    request.origin = origin;
                    request.tool_job_id = tool_job_id;
                }
                std::string question = trim_ascii_copy(json_value(arguments, "question", std::string()));
                if (question.empty()) {
                    question = trim_ascii_copy(intent_hint);
                }
                const json options_json = arguments.value("options", json::array());
                std::vector<std::string> options;
                if (options_json.is_array()) {
                    for (const auto & item : options_json) {
                        const std::string label = trim_ascii_copy(item.is_string() ? item.get<std::string>() : std::string());
                        if (!label.empty()) {
                            options.push_back(label);
                        }
                        if ((int32_t) options.size() >= LLAMA_TELEGRAM_ASK_MAX_OPTIONS) {
                            break;
                        }
                    }
                }
                if (question.empty() || options.size() < 2) {
                    if (out_error) {
                        *out_error = "ask_with_options tool call requires a question and at least two options";
                    }
                    return false;
                }
                if (user_facing_text_violates_plain_prose_policy(question)) {
                    if (out_error) {
                        *out_error = "ask_with_options question violated the plain-prose policy";
                    }
                    return false;
                }
                for (const auto & option : options) {
                    if (user_facing_text_violates_plain_prose_policy(option)) {
                        if (out_error) {
                            *out_error = "ask_with_options option violated the plain-prose policy";
                        }
                        return false;
                    }
                }

                const std::string dedupe_key = trim_ascii_copy(json_value(arguments, "dedupeKey", std::string()));
                const std::string chat_scope =
                        !trim_ascii_copy(task.telegram_chat_scope).empty() ?
                                trim_ascii_copy(task.telegram_chat_scope) :
                                dmn_telegram_dialogue_scope_locked();
                if (chat_scope.empty()) {
                    if (out_error) {
                        *out_error = "ask_with_options could not resolve a Telegram chat scope";
                    }
                    return false;
                }

                request.urgency = std::clamp(
                        json_value(
                                arguments,
                                "urgency",
                                task.has_dmn_trace ? task.dmn_trace.winner_score : 0.5f),
                        0.0f,
                        1.0f);
                request.command_ready = true;
                request.option_count = std::min<int32_t>((int32_t) options.size(), LLAMA_TELEGRAM_ASK_MAX_OPTIONS);
                std::snprintf(request.question, sizeof(request.question), "%s", question.c_str());
                std::snprintf(request.chat_scope, sizeof(request.chat_scope), "%s", chat_scope.c_str());
                std::snprintf(
                        request.dedupe_key,
                        sizeof(request.dedupe_key),
                        "%s",
                        dedupe_key.empty() ? ("react-ask-" + std::to_string(command_id)).c_str() : dedupe_key.c_str());
                for (int32_t i = 0; i < request.option_count; ++i) {
                    std::snprintf(request.options[i].label, sizeof(request.options[i].label), "%s", options[(size_t) i].c_str());
                }
                if (llama_cognitive_telegram_ask_options_set_request(ctx, &request) != 0) {
                    if (out_error) {
                        *out_error = "failed to install updated ask_with_options request";
                    }
                    return false;
                }
            } else {
                llama_telegram_relay_request request = {};
                if (llama_cognitive_telegram_relay_get_request(ctx, command_id, &request) != 0) {
                    request.command_id = command_id;
                    request.origin = origin;
                    request.tool_job_id = tool_job_id;
                }
                std::string relay_text = trim_ascii_copy(json_value(arguments, "text", std::string()));
                if (relay_text.empty()) {
                    relay_text = trim_ascii_copy(intent_hint);
                }
                if (relay_text.empty()) {
                    if (out_error) {
                        *out_error = "telegram relay tool call did not include any text";
                    }
                    return false;
                }
                if (user_facing_text_violates_plain_prose_policy(relay_text)) {
                    if (out_error) {
                        *out_error = "telegram relay text violated the plain-prose policy";
                    }
                    return false;
                }
                const std::string dedupe_key = trim_ascii_copy(json_value(arguments, "dedupeKey", std::string()));
                const std::string intent = trim_ascii_copy(json_value(arguments, "intent", std::string()));
                request.intent_kind = telegram_intent_kind_from_text(intent);
                request.urgency = std::clamp(
                        json_value(
                                arguments,
                                "urgency",
                                task.has_dmn_trace ? task.dmn_trace.winner_score : 0.5f),
                        0.0f,
                        1.0f);
                request.command_ready = true;
                std::snprintf(
                        request.dedupe_key,
                        sizeof(request.dedupe_key),
                        "%s",
                        dedupe_key.empty() ? ("react-relay-" + std::to_string(command_id)).c_str() : dedupe_key.c_str());
                std::snprintf(request.text, sizeof(request.text), "%s", relay_text.c_str());
                if (llama_cognitive_telegram_relay_set_request(ctx, &request) != 0) {
                    if (out_error) {
                        *out_error = "failed to install updated telegram relay request";
                    }
                    return false;
                }
            }
        } else {
            if (out_error) {
                *out_error = "selected capability does not have a supported dispatch backend";
            }
            return false;
        }

        SRV_INF("react rebound command=%d tool=\"%s\" backend=%d capability=\"%s\"\n",
                command_id,
                emitted_tool_name.empty() ? capability->descriptor.tool_name.c_str() : emitted_tool_name.c_str(),
                (int) capability->backend,
                capability->descriptor.capability_id.c_str());
        return true;
    }

    void append_react_tool_result(server_task & task, const vicuna_external_work_result & result) const {
        std::string tool_observation;
        if (result.kind == VICUNA_EXTERNAL_WORK_HARD_MEMORY) {
            tool_observation = format_react_hard_memory_observation(result.hard_memory_result);
        } else if (result.kind == VICUNA_EXTERNAL_WORK_CODEX) {
            tool_observation = format_react_codex_observation(result.codex_result);
        } else {
            tool_observation = format_react_bash_observation(result.bash_result);
        }
        task.react_last_tool_observation = react_limit_turn_state_text(tool_observation);
        append_react_history_entry(
                task,
                "tool_result",
                "tool_result",
                task.react_last_tool_observation);
        record_react_last_step(
                task,
                "tool_result",
                react_json_text({
                        {"tool_family_id", task.react_last_tool_family_id},
                        {"method_name", task.react_last_tool_method_name},
                        {"capability_id", task.react_last_tool_capability_id},
                }),
                task.react_last_tool_observation);
        const int32_t source_id =
                task.react_origin == SERVER_REACT_ORIGIN_DMN ?
                        task.dmn_trace.tick_id :
                        task.active_trace.episode_id;
        const int32_t plan_id =
                task.react_origin == SERVER_REACT_ORIGIN_DMN ?
                        task.dmn_trace.plan.plan_id :
                        task.active_trace.plan.plan_id;
        (void) admit_runtime_emit_text(
                ctx,
                tool_observation,
                result.origin,
                LLAMA_COG_LOOP_PHASE_OBSERVE,
                source_id,
                plan_id,
                0,
                LLAMA_SELF_COG_ARTIFACT_TOOL_OBSERVATION);
    }

    void mark_runtime_state_dirty(const char * reason) const {
        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        runtime_state_dirty = true;
        if (reason && reason[0] != '\0') {
            SRV_DBG("runtime state dirty: %s\n", reason);
        }
    }

    bool tool_registry_has_codex() const {
        if (!ctx) {
            return false;
        }
        const int32_t spec_count = llama_cognitive_tool_spec_count(ctx);
        for (int32_t i = 0; i < spec_count; ++i) {
            llama_cognitive_tool_spec spec = {};
            if (llama_cognitive_tool_spec_get(ctx, i, &spec) == 0 &&
                    spec.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI) {
                return true;
            }
        }
        return false;
    }

    std::string compose_codex_completion_message(
            const std::string & base_text,
            bool rebuild_succeeded,
            bool accessibility_verified) const {
        std::string message = trim_ascii_copy(base_text);
        if (message.empty()) {
            message = "Codex completed a repository change.";
        }
        if (rebuild_succeeded) {
            message += accessibility_verified ?
                    "\n\nRebuild completed and the Codex tool is accessible in the restarted runtime." :
                    "\n\nRebuild completed, but the restarted runtime did not confirm that the Codex tool is accessible.";
        } else {
            message += "\n\nThe rebuild did not complete successfully. Inspect the runtime and rebuild logs before trusting the change.";
        }
        return message;
    }

    void drain_codex_completion_messages() {
        if (!ctx) {
            return;
        }
        llama_codex_tool_config config = llama_codex_tool_default_config();
        if (llama_codex_tool_get_config(ctx, &config) != 0 ||
                config.completion_message_path[0] == '\0') {
            return;
        }
        const std::filesystem::path message_path(config.completion_message_path);
        if (!std::filesystem::exists(message_path)) {
            return;
        }

        const std::string raw = read_text_file_bounded(message_path, 32 * 1024);
        if (raw.empty()) {
            std::error_code ec;
            std::filesystem::remove(message_path, ec);
            return;
        }

        std::error_code ec;
        std::filesystem::remove(message_path, ec);

        const bool rebuild_succeeded = raw.find("REBUILD_STATUS=ok") != std::string::npos;
        std::string base_text = raw;
        const std::string ok_prefix = "REBUILD_STATUS=ok\n";
        const std::string fail_prefix = "REBUILD_STATUS=failed\n";
        if (base_text.rfind(ok_prefix, 0) == 0) {
            base_text.erase(0, ok_prefix.size());
        } else if (base_text.rfind(fail_prefix, 0) == 0) {
            base_text.erase(0, fail_prefix.size());
        }
        const bool accessibility_verified = tool_registry_has_codex();
        const std::string message = compose_codex_completion_message(base_text, rebuild_succeeded, accessibility_verified);
        if (proactive_mailbox_publish(message, "codex-rebuild")) {
            (void) admit_runtime_emit_text(
                    ctx,
                    message,
                    LLAMA_COG_COMMAND_ORIGIN_DMN,
                    LLAMA_FUNCTIONAL_MICROPHASE_NONE,
                    -1,
                    -1,
                    LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP);
            mark_runtime_state_dirty("codex-rebuild-completion");
        }
    }

    bool schedule_codex_rebuild_after_completion(
            const llama_codex_tool_request & request,
            llama_codex_tool_result * result) {
        if (!ctx || !result) {
            return false;
        }

        if (result->launch_failed ||
                result->exit_code != 0 ||
                !result->repo_changed ||
                !request.rebuild_after_changes ||
                request.rebuild_script_path[0] == '\0' ||
                request.rebuild_helper_path[0] == '\0' ||
                request.completion_message_path[0] == '\0') {
            return true;
        }

        const std::string maintenance_text =
                "I am having work done on myself through the Codex tool and will rebuild to apply the change.";
        (void) admit_runtime_emit_text(
                ctx,
                maintenance_text,
                request.origin,
                LLAMA_FUNCTIONAL_MICROPHASE_NONE,
                request.command_id,
                -1,
                LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP);
        mark_runtime_state_dirty("codex-tool-pre-rebuild");
        (void) persist_runtime_state("codex-tool-pre-rebuild");

        std::string pending_message = trim_ascii_copy(result->summary_text);
        if (!trim_ascii_copy(result->changed_files_excerpt).empty()) {
            pending_message += "\n\nChanged files:\n";
            pending_message += trim_ascii_copy(result->changed_files_excerpt);
        }
        if (!trim_ascii_copy(result->manual_requirements).empty()) {
            pending_message += "\n\nManual requirements: ";
            pending_message += trim_ascii_copy(result->manual_requirements);
        }

        const std::filesystem::path completion_path(request.completion_message_path);
        const std::filesystem::path pending_path = completion_path.string() + ".pending";
        std::string pending_error;
        if (!write_text_file(pending_path, pending_message, &pending_error)) {
            std::snprintf(result->error_text, sizeof(result->error_text), "%s", pending_error.c_str());
            std::snprintf(result->summary_text, sizeof(result->summary_text), "%s",
                    "Codex applied changes, but the rebuild handoff file could not be written.");
            result->rebuild_attempted = false;
            result->rebuild_succeeded = false;
            return false;
        }

        std::string launch_command;
        const char * systemd_run_paths[] = { "/usr/bin/systemd-run", "/bin/systemd-run" };
        const char * systemd_run_path = nullptr;
        for (const char * candidate : systemd_run_paths) {
            if (std::filesystem::exists(candidate)) {
                systemd_run_path = candidate;
                break;
            }
        }
        if (systemd_run_path) {
            launch_command =
                    shell_quote(systemd_run_path) +
                    " --user --unit " + shell_quote("vicuna-codex-rebuild-" + std::to_string(request.tool_job_id)) +
                    " " + shell_quote(request.rebuild_helper_path) +
                    " --pending-message-file " + shell_quote(pending_path.string()) +
                    " --final-message-file " + shell_quote(request.completion_message_path) +
                    " --rebuild-script " + shell_quote(request.rebuild_script_path);
        } else {
            launch_command =
                    "nohup " + shell_quote(request.rebuild_helper_path) +
                    " --pending-message-file " + shell_quote(pending_path.string()) +
                    " --final-message-file " + shell_quote(request.completion_message_path) +
                    " --rebuild-script " + shell_quote(request.rebuild_script_path) +
                    " >/dev/null 2>&1 &";
        }

        const std::string repo_root = trim_ascii_copy(request.working_directory);
        llama_bash_tool_result launch_result = {};
        (void) run_permissive_bash_command(
                repo_root,
                launch_command,
                10000,
                LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1,
                LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1,
                &launch_result);
        result->rebuild_attempted = true;
        result->rebuild_succeeded =
                !launch_result.launch_failed && launch_result.exit_code == 0;
        if (result->rebuild_succeeded) {
            std::snprintf(result->summary_text, sizeof(result->summary_text), "%s",
                    "Codex applied repository changes after completing the tool call and scheduled a rebuild. A completion message will be emitted after restart.");
        } else {
            std::snprintf(result->summary_text, sizeof(result->summary_text), "%s",
                    "Codex applied repository changes, but scheduling the post-completion rebuild failed.");
            if (launch_result.error_text[0] != '\0') {
                std::snprintf(result->error_text, sizeof(result->error_text), "%s", launch_result.error_text);
            }
        }
        return result->rebuild_succeeded;
    }

    bool collect_progress_snapshot(
            vicuna_progress_snapshot * out_snapshot,
            llama_self_model_state_info * out_model_state = nullptr,
            llama_functional_lora_trace * out_functional_trace = nullptr,
            llama_process_functional_trace * out_process_trace = nullptr) const {
        if (!ctx || !out_snapshot) {
            return false;
        }

        llama_self_model_state_info model_state = {};
        if (llama_self_state_get_model_state(ctx, &model_state) != 0) {
            return false;
        }
        llama_functional_lora_trace functional_trace = {};
        (void) llama_functional_lora_get_last_trace(ctx, &functional_trace);
        llama_process_functional_trace process_trace = {};
        (void) llama_process_functional_get_last_trace(ctx, &process_trace);

        uint64_t functional_updates = 0;
        for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
            functional_updates += functional_trace.family_state[family].update_count;
        }

        uint64_t process_updates = 0;
        const int32_t process_entry_count = llama_process_functional_entry_count(ctx);
        for (int32_t i = 0; i < process_entry_count; ++i) {
            llama_process_functional_entry_info info = {};
            if (llama_process_functional_entry_get(ctx, i, &info) == 0 && info.valid) {
                process_updates += info.update_count;
            }
        }

        out_snapshot->valid = true;
        out_snapshot->active_extensions = model_state.extension_summary.active_count;
        out_snapshot->discovered_extensions = model_state.extension_summary.discovered_count;
        out_snapshot->permanent_extensions = model_state.extension_summary.permanent_count;
        out_snapshot->allostatic_extensions = model_state.extension_summary.allostatic_count;
        out_snapshot->allostatic_divergence = model_state.extension_summary.allostatic_divergence;
        out_snapshot->promotion_readiness = model_state.belief_summary.promotion_readiness;
        out_snapshot->belief_pressure = model_state.belief_summary.residual_allostatic_pressure;
        out_snapshot->functional_update_total = functional_updates;
        out_snapshot->process_update_total = process_updates;

        if (out_model_state) {
            *out_model_state = model_state;
        }
        if (out_functional_trace) {
            *out_functional_trace = functional_trace;
        }
        if (out_process_trace) {
            *out_process_trace = process_trace;
    }
    return true;
}

static std::string normalize_telegram_chat_scope(const std::string & value) {
    const std::string normalized = trim_ascii_copy(value);
    return normalized.empty() ? std::string() : normalized;
}

static bool telegram_dialogue_scope_is_broadcast(const std::string & scope) {
    return scope == VICUNA_TELEGRAM_DIALOGUE_SCOPE_BROADCAST;
}

static vicuna_telegram_dialogue_snapshot telegram_dialogue_snapshot_copy(
        const vicuna_telegram_dialogue_history & history) {
    std::lock_guard<std::mutex> lock(history.mutex);

    vicuna_telegram_dialogue_snapshot snapshot = {};
    snapshot.max_turns_per_scope = history.max_turns_per_scope;
    snapshot.next_turn_id = history.next_turn_id;
    snapshot.turns_by_scope = history.turns_by_scope;
    snapshot.latest_message_id_by_scope = history.latest_message_id_by_scope;
    return snapshot;
}

static json telegram_dialogue_history_to_json(const vicuna_telegram_dialogue_history & history) {
    const vicuna_telegram_dialogue_snapshot snapshot = telegram_dialogue_snapshot_copy(history);

    json scopes = json::array();
    for (const auto & [scope, turns] : snapshot.turns_by_scope) {
        json scope_turns = json::array();
        for (const auto & turn : turns) {
            scope_turns.push_back({
                {"turn_id", turn.turn_id},
                {"chat_scope", turn.chat_scope},
                {"user_text", turn.user_text},
                {"assistant_text", turn.assistant_text},
                {"assistant_source", turn.assistant_source},
                {"response_id", turn.response_id},
                {"dedupe_key", turn.dedupe_key},
                {"telegram_message_id", turn.telegram_message_id},
                {"updated_at_ms", turn.updated_at_ms},
            });
        }
        const auto latest_it = snapshot.latest_message_id_by_scope.find(scope);
        scopes.push_back({
            {"chat_scope", scope},
            {"latest_message_id", latest_it != snapshot.latest_message_id_by_scope.end() ? latest_it->second : int64_t(0)},
            {"turns", scope_turns},
        });
    }

    return json {
        {"max_turns_per_scope", snapshot.max_turns_per_scope},
        {"next_turn_id", snapshot.next_turn_id},
        {"scopes", scopes},
    };
}

static bool telegram_dialogue_history_from_json(
        const json & data,
        vicuna_telegram_dialogue_history * out_history) {
    if (!out_history || !data.is_object()) {
        return false;
    }

    vicuna_telegram_dialogue_snapshot restored = {};
    restored.max_turns_per_scope = std::max<size_t>(
            1,
            json_value(data, "max_turns_per_scope", restored.max_turns_per_scope));
    restored.next_turn_id = std::max<uint64_t>(1, json_value(data, "next_turn_id", restored.next_turn_id));

    if (data.contains("scopes") && data.at("scopes").is_array()) {
        for (const auto & scope_entry : data.at("scopes")) {
            if (!scope_entry.is_object()) {
                continue;
            }
            const std::string scope = normalize_telegram_chat_scope(
                    json_value(scope_entry, "chat_scope", std::string()));
            if (scope.empty()) {
                continue;
            }

            std::deque<vicuna_telegram_dialogue_turn> turns;
            if (scope_entry.contains("turns") && scope_entry.at("turns").is_array()) {
                for (const auto & turn_entry : scope_entry.at("turns")) {
                    if (!turn_entry.is_object()) {
                        continue;
                    }
                    vicuna_telegram_dialogue_turn turn = {};
                    turn.turn_id = std::max<uint64_t>(1, json_value(turn_entry, "turn_id", uint64_t(0)));
                    turn.chat_scope = scope;
                    turn.user_text = trim_ascii_copy(json_value(turn_entry, "user_text", std::string()));
                    turn.assistant_text = trim_ascii_copy(json_value(turn_entry, "assistant_text", std::string()));
                    turn.assistant_source = trim_ascii_copy(json_value(turn_entry, "assistant_source", std::string()));
                    turn.response_id = trim_ascii_copy(json_value(turn_entry, "response_id", std::string()));
                    turn.dedupe_key = trim_ascii_copy(json_value(turn_entry, "dedupe_key", std::string()));
                    turn.telegram_message_id = json_value(turn_entry, "telegram_message_id", int64_t(0));
                    turn.updated_at_ms = json_value(turn_entry, "updated_at_ms", int64_t(0));
                    if (turn.user_text.empty() && turn.assistant_text.empty()) {
                        continue;
                    }
                    turns.push_back(std::move(turn));
                }
            }

            if (!turns.empty()) {
                while (turns.size() > restored.max_turns_per_scope) {
                    turns.pop_front();
                }
                restored.turns_by_scope[scope] = std::move(turns);
            }

            const int64_t latest_message_id = json_value(scope_entry, "latest_message_id", int64_t(0));
            if (latest_message_id > 0) {
                restored.latest_message_id_by_scope[scope] = latest_message_id;
            }
        }
    }

    std::lock_guard<std::mutex> lock(out_history->mutex);
    out_history->max_turns_per_scope = restored.max_turns_per_scope;
    out_history->next_turn_id = restored.next_turn_id;
    out_history->turns_by_scope = std::move(restored.turns_by_scope);
    out_history->latest_message_id_by_scope = std::move(restored.latest_message_id_by_scope);
    return true;
}

    void note_provenance_progress_locked(const vicuna_progress_snapshot & snapshot, const std::string & event_kind) {
        if (!snapshot.valid) {
            return;
        }

        if (event_kind == "active_loop") {
            provenance_repository.active_loop_total++;
        } else if (event_kind == "tool_result") {
            provenance_repository.tool_result_total++;
        } else if (event_kind == "dmn_tick") {
            provenance_repository.dmn_total++;
        }

        if (provenance_repository.has_last_snapshot) {
            const auto & prev = provenance_repository.last_snapshot;
            if (snapshot.discovered_extensions > prev.discovered_extensions) {
                provenance_repository.discovered_increase_total +=
                        (uint64_t) (snapshot.discovered_extensions - prev.discovered_extensions);
            }
            if (snapshot.permanent_extensions > prev.permanent_extensions) {
                provenance_repository.permanent_increase_total +=
                        (uint64_t) (snapshot.permanent_extensions - prev.permanent_extensions);
            }
            if (snapshot.allostatic_extensions > prev.allostatic_extensions) {
                provenance_repository.allostatic_increase_total +=
                        (uint64_t) (snapshot.allostatic_extensions - prev.allostatic_extensions);
            }
            if (snapshot.functional_update_total > prev.functional_update_total) {
                provenance_repository.functional_update_observed_total +=
                        snapshot.functional_update_total - prev.functional_update_total;
            }
            if (snapshot.process_update_total > prev.process_update_total) {
                provenance_repository.process_update_observed_total +=
                        snapshot.process_update_total - prev.process_update_total;
            }
        }

        provenance_repository.last_snapshot = snapshot;
        provenance_repository.has_last_snapshot = true;
    }

    static std::string format_provenance_bucket_suffix_utc(int64_t timestamp_ms) {
        const std::time_t seconds = (std::time_t) std::max<int64_t>(0, timestamp_ms / 1000);
        std::tm broken_down = {};
#if defined(_WIN32)
        gmtime_s(&broken_down, &seconds);
#else
        gmtime_r(&seconds, &broken_down);
#endif
        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y%m%dT%H", &broken_down);
        return buffer;
    }

    static std::filesystem::path provenance_segment_path_for(
            const std::string & configured_path,
            int64_t timestamp_ms) {
        const std::filesystem::path base_path(configured_path);
        const std::string stem = base_path.stem().string();
        const std::string extension = base_path.extension().string();
        const std::string suffix = format_provenance_bucket_suffix_utc(timestamp_ms);
        const std::string filename =
                extension.empty() ?
                        stem + "." + suffix :
                        stem + "." + suffix + extension;
        return base_path.parent_path() / filename;
    }

    static bool provenance_segment_matches_base(
            const std::filesystem::path & candidate,
            const std::filesystem::path & configured_base) {
        const std::string stem = configured_base.stem().string();
        const std::string extension = configured_base.extension().string();
        const std::string filename = candidate.filename().string();
        const std::string prefix = stem + ".";
        if (!string_starts_with(filename, prefix)) {
            return false;
        }
        if (!extension.empty() && !string_ends_with(filename, extension)) {
            return false;
        }
        return true;
    }

    std::string resolve_provenance_segment_path_locked(int64_t now_ms) {
        const int64_t bucket_ms = 60LL * 60LL * 1000LL;
        const int64_t bucket_start = (now_ms / bucket_ms) * bucket_ms;
        if (!provenance_repository.active_path.empty() &&
                provenance_repository.active_bucket_start_ms == bucket_start) {
            return provenance_repository.active_path;
        }
        provenance_repository.active_bucket_start_ms = bucket_start;
        provenance_repository.active_path =
                provenance_segment_path_for(provenance_repository.path, now_ms).string();
        return provenance_repository.active_path;
    }

    bool prune_provenance_segments(
            const std::string & configured_path,
            const std::string & active_path,
            int64_t retention_ms,
            uint64_t * out_pruned_count,
            std::string * out_error) const {
        if (out_pruned_count) {
            *out_pruned_count = 0;
        }
        if (configured_path.empty()) {
            return true;
        }
        try {
            const std::filesystem::path configured_base(configured_path);
            const std::filesystem::path parent =
                    configured_base.parent_path().empty() ? std::filesystem::current_path() : configured_base.parent_path();
            if (!std::filesystem::exists(parent)) {
                return true;
            }
            const auto cutoff =
                    std::filesystem::file_time_type::clock::now() - std::chrono::milliseconds(std::max<int64_t>(retention_ms, 0));
            uint64_t pruned = 0;
            for (const auto & entry : std::filesystem::directory_iterator(parent)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const std::filesystem::path candidate = entry.path();
                if (candidate == std::filesystem::path(active_path)) {
                    continue;
                }
                if (!provenance_segment_matches_base(candidate, configured_base) &&
                        candidate != configured_base) {
                    continue;
                }
                std::error_code time_ec;
                const auto last_write = std::filesystem::last_write_time(candidate, time_ec);
                if (time_ec || last_write >= cutoff) {
                    continue;
                }
                std::error_code remove_ec;
                if (std::filesystem::remove(candidate, remove_ec) && !remove_ec) {
                    ++pruned;
                }
            }
            if (out_pruned_count) {
                *out_pruned_count = pruned;
            }
            return true;
        } catch (const std::exception & err) {
            if (out_error) {
                *out_error = err.what();
            }
            return false;
        }
    }

    bool append_provenance_event(
            const std::string & event_kind,
            const std::string & source,
            const json & payload,
            const vicuna_progress_snapshot & snapshot) {
        std::string path;
        std::string configured_path;
        int64_t retention_ms = 0;
        json event = json::object();
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            if (!provenance_repository.enabled || provenance_repository.path.empty()) {
                return false;
            }
            configured_path = provenance_repository.path;
            path = resolve_provenance_segment_path_locked(ggml_time_ms());
            retention_ms = provenance_repository.retention_ms;
            event["schema_version"] = 1;
            event["session_id"] = provenance_repository.session_id;
            event["sequence"] = provenance_repository.next_sequence++;
            event["timestamp_ms"] = ggml_time_ms();
            event["event_kind"] = event_kind;
            event["source"] = source;
            event["repository_path"] = configured_path;
            event["segment_path"] = path;
            event["payload"] = payload;
        }

        try {
            const std::filesystem::path target_path(path);
            if (!target_path.parent_path().empty()) {
                std::filesystem::create_directories(target_path.parent_path());
            }
            std::ofstream out(target_path, std::ios::binary | std::ios::app);
            if (!out) {
                throw std::runtime_error("failed to open provenance repository");
            }
            out << event.dump() << '\n';
            out.flush();
            if (!out) {
                throw std::runtime_error("failed to flush provenance repository");
            }
        } catch (const std::exception & err) {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            provenance_repository.healthy = false;
            provenance_repository.append_fail_total++;
            provenance_repository.last_error = err.what();
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            provenance_repository.healthy = true;
            provenance_repository.append_total++;
            provenance_repository.last_append_ms = ggml_time_ms();
            provenance_repository.last_error.clear();
            note_provenance_progress_locked(snapshot, event_kind);
        }

        uint64_t pruned_count = 0;
        std::string prune_error;
        if (!prune_provenance_segments(configured_path, path, retention_ms, &pruned_count, &prune_error)) {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            provenance_repository.last_error = prune_error;
        } else if (pruned_count > 0) {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            provenance_repository.prune_total += pruned_count;
        }
        return true;
    }

    void capture_active_loop_provenance(
            const char * source,
            const llama_active_loop_trace & active_trace,
            const json * extra = nullptr) {
        vicuna_progress_snapshot progress = {};
        llama_self_model_state_info model_state = {};
        llama_functional_lora_trace functional_trace = {};
        llama_process_functional_trace process_trace = {};
        if (!collect_progress_snapshot(&progress, &model_state, &functional_trace, &process_trace)) {
            return;
        }

        json payload = {
            {"active_loop", active_trace_summary_to_json(active_trace)},
            {"self_model", self_model_summary_to_json(model_state)},
            {"functional", functional_trace_summary_to_json(functional_trace)},
            {"process_functional", process_trace_summary_to_json(process_trace)},
        };
        if (extra) {
            payload["extra"] = *extra;
        }
        (void) append_provenance_event("active_loop", source ? source : "active", payload, progress);
    }

    json command_authorization_trace_json(
            const llama_cognitive_command & command,
            const server_openclaw_capability_runtime * capability,
            const vicuna_pending_approval * approval = nullptr) const {
        json trace = {
            {"resolved_execution_safety_class", command_execution_safety_class(command, capability)},
        };
        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        trace["authorization_source"] = command_authorization_source_locked(command, capability, approval);
        if (const vicuna_foreground_consent_scope * scope = foreground_consent_scope_for_command_locked(command)) {
            trace["foreground_consent"] = {
                {"scope_id", scope->scope_id},
                {"episode_id", scope->episode_id},
                {"task_id", scope->task_id},
            };
        }
        return trace;
    }

    json active_waiting_task_provenance_extra(int32_t command_id, const llama_active_loop_trace * active_trace = nullptr) {
        json extra = json::object();
        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        auto it = waiting_active_tasks.find(command_id);
        if (it == waiting_active_tasks.end()) {
            return extra;
        }
        const server_task & task = it->second;
        if (task.foreground_consent_active) {
            extra["foreground_consent"] = {
                {"requested", task.foreground_consent_requested},
                {"active", task.foreground_consent_active},
                {"episode_id", task.foreground_consent_episode_id},
                {"scope_id", task.foreground_consent_scope_id},
            };
        }
        if (active_trace) {
            extra["active_episode_id"] = active_trace->episode_id;
        }
        return extra;
    }

    void capture_tool_call_provenance(
            const char * source,
            const llama_cognitive_command & command,
            const json & tool_call,
            const json * extra = nullptr) {
        vicuna_progress_snapshot progress = {};
        llama_self_model_state_info model_state = {};
        llama_functional_lora_trace functional_trace = {};
        llama_process_functional_trace process_trace = {};
        if (!collect_progress_snapshot(&progress, &model_state, &functional_trace, &process_trace)) {
            return;
        }

        json payload = {
            {"command", {
                {"command_id", command.command_id},
                {"origin", command.origin},
                {"kind", command.kind},
                {"status", command.status},
                {"episode_id", command.episode_id},
                {"tick_id", command.tick_id},
                {"tool_kind", command.tool_kind},
                {"tool_spec_index", command.tool_spec_index},
                {"tool_job_id", command.tool_job_id},
                {"reason_mask", command.reason_mask},
                {"priority", command.priority},
                {"source_family", command.source_family},
                {"loop_phase", command.loop_phase},
                {"capability_id", bounded_cstr_to_string(command.capability_id)},
            }},
            {"tool_call", tool_call},
            {"self_model", self_model_summary_to_json(model_state)},
            {"functional", functional_trace_summary_to_json(functional_trace)},
            {"process_functional", process_trace_summary_to_json(process_trace)},
        };

        if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
            llama_active_loop_trace active_trace = {};
            const json waiting_extra = active_waiting_task_provenance_extra(command.command_id);
            if (llama_active_loop_get_last_trace(ctx, &active_trace) == 0) {
                payload["active_loop"] = active_trace_summary_to_json(active_trace);
            }
            if (!waiting_extra.empty()) {
                payload["extra"] = waiting_extra;
            }
        } else if (command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
            llama_dmn_tick_trace dmn_trace = {};
            if (llama_dmn_get_last_trace(ctx, &dmn_trace) == 0) {
                payload["dmn"] = dmn_trace_summary_to_json(dmn_trace);
            }
        }

        if (extra && !extra->empty()) {
            if (payload.contains("extra") && payload["extra"].is_object()) {
                for (auto it = extra->begin(); it != extra->end(); ++it) {
                    payload["extra"][it.key()] = it.value();
                }
            } else {
                payload["extra"] = *extra;
            }
        }

        (void) append_provenance_event("tool_call", source ? source : "tool", payload, progress);
    }

    void capture_tool_result_provenance(
            const char * source,
            const json & tool_result,
            const llama_active_loop_trace * active_trace = nullptr) {
        vicuna_progress_snapshot progress = {};
        llama_self_model_state_info model_state = {};
        llama_functional_lora_trace functional_trace = {};
        llama_process_functional_trace process_trace = {};
        if (!collect_progress_snapshot(&progress, &model_state, &functional_trace, &process_trace)) {
            return;
        }

        json payload = {
            {"tool_result", tool_result},
            {"self_model", self_model_summary_to_json(model_state)},
            {"functional", functional_trace_summary_to_json(functional_trace)},
            {"process_functional", process_trace_summary_to_json(process_trace)},
        };
        if (active_trace) {
            payload["active_loop"] = active_trace_summary_to_json(*active_trace);
        }
        (void) append_provenance_event("tool_result", source ? source : "tool", payload, progress);
    }

    void capture_dmn_provenance(const char * source, const llama_dmn_tick_trace & dmn_trace) {
        vicuna_progress_snapshot progress = {};
        llama_self_model_state_info model_state = {};
        llama_functional_lora_trace functional_trace = {};
        llama_process_functional_trace process_trace = {};
        if (!collect_progress_snapshot(&progress, &model_state, &functional_trace, &process_trace)) {
            return;
        }

        llama_counterfactual_trace counterfactual = {};
        llama_governance_trace governance = {};
        llama_remediation_plan remediation = {};
        llama_temporal_self_improvement_trace temporal = {};
        (void) llama_counterfactual_get_last_trace(ctx, &counterfactual);
        (void) llama_governance_get_last_trace(ctx, &governance);
        (void) llama_remediation_get_last_plan(ctx, &remediation);
        (void) llama_temporal_self_improvement_get_last(ctx, &temporal);

        json payload = {
            {"dmn", dmn_trace_summary_to_json(dmn_trace)},
            {"counterfactual", counterfactual_trace_summary_to_json(counterfactual)},
            {"governance", governance_trace_summary_to_json(governance)},
            {"remediation", remediation_plan_to_json(remediation)},
            {"temporal_self_improvement", {
                {"valid", temporal.valid},
                {"loop_origin", temporal.loop_origin},
                {"selected_temporal_role", temporal.selected_temporal_role},
                {"counterfactual_family", temporal.counterfactual_family},
                {"outcome", temporal.outcome},
                {"signed_advantage", temporal.signed_advantage},
                {"efficiency_advantage", temporal.efficiency_advantage},
                {"evolution_uncertainty_before", temporal.evolution_uncertainty_before},
                {"evolution_uncertainty_after", temporal.evolution_uncertainty_after},
            }},
            {"self_model", self_model_summary_to_json(model_state)},
            {"functional", functional_trace_summary_to_json(functional_trace)},
            {"process_functional", process_trace_summary_to_json(process_trace)},
        };
        (void) append_provenance_event("dmn_tick", source ? source : "dmn", payload, progress);
    }

    void post_next_response_task() {
        server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
        task.id = queue_tasks.get_new_id();
        queue_tasks.post(std::move(task), true);
    }

    static void proactive_mailbox_rebuild_live_events_locked(vicuna_proactive_mailbox & mailbox) {
        mailbox.live_events.clear();
        for (const std::string & response_id : mailbox.response_order) {
            auto it = mailbox.responses.find(response_id);
            if (it == mailbox.responses.end()) {
                continue;
            }
            for (const auto & event : it->second.events) {
                mailbox.live_events.push_back(event);
            }
        }
    }

    static void proactive_mailbox_prune_locked(vicuna_proactive_mailbox & mailbox) {
        bool pruned = false;
        while (mailbox.response_order.size() > mailbox.max_responses) {
            const std::string response_id = mailbox.response_order.front();
            mailbox.response_order.pop_front();
            if (mailbox.responses.erase(response_id) > 0) {
                mailbox.dropped_total++;
                pruned = true;
                SRV_INF("proactive mailbox evicted response=%s\n", response_id.c_str());
            }
        }
        if (pruned) {
            proactive_mailbox_rebuild_live_events_locked(mailbox);
        }
    }

    bool proactive_mailbox_publish(
            const std::string & text,
            const char * source,
            std::string * out_response_id = nullptr) const {
        const std::string trimmed = trim_ascii_copy(text);
        if (trimmed.empty()) {
            return false;
        }

        const std::string response_id = "resp_" + random_string();
        const std::string message_id = "msg_" + random_string();
        const std::time_t now_seconds = std::time(nullptr);
        const int64_t now_ms = ggml_time_ms();

        vicuna_stored_response stored = {};
        stored.response_id = response_id;
        stored.created_ms = now_ms;
        stored.completed_ms = now_ms;
        stored.response = build_proactive_response_object(
                response_id,
                message_id,
                model_name.empty() ? "vicuna" : model_name,
                trimmed,
                now_seconds);

        std::vector<json> events = build_proactive_response_events(
                response_id,
                message_id,
                trimmed,
                stored.response);

        {
            std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
            for (const auto & event : events) {
                vicuna_mailbox_event stored_event = {};
                stored_event.sequence_number = proactive_mailbox.next_sequence_number++;
                stored_event.response_id = response_id;
                stored_event.event = event;
                proactive_mailbox.live_events.push_back(stored_event);
                stored.events.push_back(std::move(stored_event));
            }
            proactive_mailbox.responses[response_id] = stored;
            proactive_mailbox.response_order.push_back(response_id);
            proactive_mailbox.publish_total++;
            proactive_mailbox.complete_total++;
            proactive_mailbox.last_publish_ms = now_ms;
            proactive_mailbox_prune_locked(proactive_mailbox);
        }

        proactive_mailbox.cv.notify_all();
        if (out_response_id) {
            *out_response_id = response_id;
        }
        SRV_INF("proactive mailbox published response=%s source=%s\n",
                response_id.c_str(),
                source ? source : "unknown");
        return true;
    }

    bool proactive_mailbox_get_response(const std::string & response_id, json * out_response) const {
        std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
        auto it = proactive_mailbox.responses.find(response_id);
        if (it == proactive_mailbox.responses.end()) {
            return false;
        }
        if (out_response) {
            *out_response = it->second.response;
        }
        return true;
    }

    bool proactive_mailbox_get_response_events(
            const std::string & response_id,
            uint64_t after_sequence,
            std::vector<json> * out_events) const {
        std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
        auto it = proactive_mailbox.responses.find(response_id);
        if (it == proactive_mailbox.responses.end()) {
            return false;
        }
        if (!out_events) {
            return true;
        }
        out_events->clear();
        for (const auto & event : it->second.events) {
            if (event.sequence_number > after_sequence) {
                out_events->push_back(event.event);
            }
        }
        return true;
    }

    std::vector<json> proactive_mailbox_collect_live_events(uint64_t after_sequence, uint64_t * out_last_sequence) const {
        std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
        std::vector<json> events;
        uint64_t last_sequence = after_sequence;
        for (const auto & event : proactive_mailbox.live_events) {
            if (event.sequence_number <= after_sequence) {
                continue;
            }
            events.push_back(event.event);
            last_sequence = event.sequence_number;
        }
        if (out_last_sequence) {
            *out_last_sequence = last_sequence;
        }
        return events;
    }

    void proactive_mailbox_wait_for_live_events(uint64_t after_sequence, const std::function<bool()> & should_stop) const {
        std::unique_lock<std::mutex> lock(proactive_mailbox.mutex);
        proactive_mailbox.cv.wait_for(
                lock,
                std::chrono::milliseconds(250),
                [&]() {
                    return should_stop() ||
                           !proactive_mailbox.live_stream_connected ||
                           (!proactive_mailbox.live_events.empty() &&
                            proactive_mailbox.live_events.back().sequence_number > after_sequence);
                });
    }

    bool proactive_mailbox_connect_live_stream() const {
        std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
        if (proactive_mailbox.live_stream_connected) {
            return false;
        }
        proactive_mailbox.live_stream_connected = true;
        SRV_INF("%s", "proactive mailbox live stream attached\n");
        return true;
    }

    void proactive_mailbox_disconnect_live_stream() const {
        bool disconnected = false;
        {
            std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
            if (proactive_mailbox.live_stream_connected) {
                proactive_mailbox.live_stream_connected = false;
                disconnected = true;
            }
        }
        if (disconnected) {
            proactive_mailbox.cv.notify_all();
            SRV_INF("%s", "proactive mailbox live stream detached\n");
        }
    }

    void proactive_mailbox_reset_live_stream() {
        {
            std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
            proactive_mailbox.live_stream_connected = false;
        }
        proactive_mailbox.cv.notify_all();
    }

    json build_proactive_mailbox_health_json() const {
        const vicuna_mailbox_snapshot snapshot = proactive_mailbox_snapshot_copy(proactive_mailbox);
        return {
            {"stored_responses", (int) snapshot.response_order.size()},
            {"stored_live_events", (int) snapshot.live_events.size()},
            {"live_stream_connected", snapshot.live_stream_connected},
            {"publish_total", snapshot.publish_total},
            {"complete_total", snapshot.complete_total},
            {"fail_total", snapshot.fail_total},
            {"dropped_total", snapshot.dropped_total},
            {"last_publish_ms", snapshot.last_publish_ms},
        };
    }

    bool telegram_outbox_publish_ask_options(
            const std::string & chat_scope,
            const std::string & dedupe_key,
            const std::string & question,
            const std::vector<std::string> & options) const {
        const std::string normalized_scope = normalize_telegram_chat_scope(chat_scope);
        const std::string trimmed_question = trim_ascii_copy(question);
        if (normalized_scope.empty() || trimmed_question.empty() || options.size() < 2) {
            return false;
        }

        vicuna_telegram_outbox_item item = {};
        item.sequence_number = 0;
        item.kind = "ask_with_options";
        item.chat_scope = normalized_scope;
        item.dedupe_key = trim_ascii_copy(dedupe_key);
        item.question = trimmed_question;
        item.options.reserve(options.size());
        for (const auto & option : options) {
            const std::string trimmed = trim_ascii_copy(option);
            if (!trimmed.empty()) {
                item.options.push_back(trimmed);
            }
        }
        if (item.options.size() < 2) {
            return false;
        }
        item.created_at_ms = ggml_time_ms();

        {
            std::lock_guard<std::mutex> lock(telegram_outbox.mutex);
            if (!item.dedupe_key.empty()) {
                for (const auto & existing : telegram_outbox.items) {
                    if (existing.kind == item.kind &&
                            existing.chat_scope == item.chat_scope &&
                            existing.dedupe_key == item.dedupe_key) {
                        return true;
                    }
                }
            }
            item.sequence_number = telegram_outbox.next_sequence_number++;
            telegram_outbox.items.push_back(std::move(item));
            telegram_outbox.publish_total++;
            telegram_outbox.last_publish_ms = ggml_time_ms();
            while (telegram_outbox.items.size() > telegram_outbox.max_items) {
                telegram_outbox.items.pop_front();
                telegram_outbox.dropped_total++;
            }
        }

        return true;
    }

    bool telegram_outbox_publish_approval_request(
            const std::string & approval_id,
            const std::string & chat_scope,
            const std::string & dedupe_key,
            const std::string & question,
            const std::vector<std::string> & options,
            int64_t reply_to_message_id,
            uint64_t * out_sequence_number = nullptr) const {
        const std::string normalized_scope = normalize_telegram_chat_scope(chat_scope);
        const std::string trimmed_question = trim_ascii_copy(question);
        if (trim_ascii_copy(approval_id).empty() ||
                normalized_scope.empty() ||
                trimmed_question.empty() ||
                options.size() < 2) {
            return false;
        }

        vicuna_telegram_outbox_item item = {};
        item.sequence_number = 0;
        item.kind = "approval_request";
        item.approval_id = trim_ascii_copy(approval_id);
        item.chat_scope = normalized_scope;
        item.dedupe_key = trim_ascii_copy(dedupe_key);
        item.reply_to_message_id = std::max<int64_t>(0, reply_to_message_id);
        item.question = trimmed_question;
        item.options.reserve(options.size());
        for (const auto & option : options) {
            const std::string trimmed = trim_ascii_copy(option);
            if (!trimmed.empty()) {
                item.options.push_back(trimmed);
            }
        }
        if (item.options.size() < 2) {
            return false;
        }
        item.created_at_ms = ggml_time_ms();

        {
            std::lock_guard<std::mutex> lock(telegram_outbox.mutex);
            if (!item.dedupe_key.empty()) {
                for (const auto & existing : telegram_outbox.items) {
                    if (existing.kind == item.kind &&
                            existing.chat_scope == item.chat_scope &&
                            existing.dedupe_key == item.dedupe_key) {
                        if (out_sequence_number) {
                            *out_sequence_number = existing.sequence_number;
                        }
                        return true;
                    }
                }
            }
            item.sequence_number = telegram_outbox.next_sequence_number++;
            telegram_outbox.items.push_back(std::move(item));
            telegram_outbox.publish_total++;
            telegram_outbox.last_publish_ms = ggml_time_ms();
            while (telegram_outbox.items.size() > telegram_outbox.max_items) {
                telegram_outbox.items.pop_front();
                telegram_outbox.dropped_total++;
            }
            if (out_sequence_number) {
                *out_sequence_number = telegram_outbox.items.back().sequence_number;
            }
        }

        return true;
    }

    bool telegram_outbox_publish_message(
            const std::string & chat_scope,
            const std::string & dedupe_key,
            const std::string & text,
            int64_t reply_to_message_id = 0) const {
        const std::string normalized_scope = normalize_telegram_chat_scope(chat_scope);
        const std::string trimmed_text = trim_ascii_copy(text);
        if (normalized_scope.empty() || trimmed_text.empty()) {
            return false;
        }

        vicuna_telegram_outbox_item item = {};
        item.sequence_number = 0;
        item.kind = "message";
        item.chat_scope = normalized_scope;
        item.dedupe_key = trim_ascii_copy(dedupe_key);
        item.text = trimmed_text;
        item.reply_to_message_id = std::max<int64_t>(0, reply_to_message_id);
        item.created_at_ms = ggml_time_ms();

        {
            std::lock_guard<std::mutex> lock(telegram_outbox.mutex);
            if (!item.dedupe_key.empty()) {
                for (const auto & existing : telegram_outbox.items) {
                    if (existing.kind == item.kind &&
                            existing.chat_scope == item.chat_scope &&
                            existing.dedupe_key == item.dedupe_key) {
                        return true;
                    }
                }
            }
            item.sequence_number = telegram_outbox.next_sequence_number++;
            telegram_outbox.items.push_back(std::move(item));
            telegram_outbox.publish_total++;
            telegram_outbox.last_publish_ms = ggml_time_ms();
            while (telegram_outbox.items.size() > telegram_outbox.max_items) {
                telegram_outbox.items.pop_front();
                telegram_outbox.dropped_total++;
            }
        }

        return true;
    }

    std::vector<json> telegram_outbox_collect_items(uint64_t after_sequence, uint64_t * out_last_sequence) const {
        std::lock_guard<std::mutex> lock(telegram_outbox.mutex);
        std::vector<json> items;
        uint64_t last_sequence = after_sequence;
        for (const auto & item : telegram_outbox.items) {
            if (item.sequence_number <= after_sequence) {
                continue;
            }
            items.push_back({
                {"sequence_number", item.sequence_number},
                {"kind", item.kind},
                {"approval_id", item.approval_id},
                {"chat_scope", item.chat_scope},
                {"dedupe_key", item.dedupe_key},
                {"text", item.text},
                {"reply_to_message_id", item.reply_to_message_id},
                {"question", item.question},
                {"options", item.options},
                {"created_at_ms", item.created_at_ms},
            });
            last_sequence = item.sequence_number;
        }
        if (out_last_sequence) {
            *out_last_sequence = last_sequence;
        }
        return items;
    }

    json submit_telegram_approval_decision(const json & body) {
        const std::string approval_id = trim_ascii_copy(json_value(body, "approval_id", std::string()));
        const std::string chat_scope = normalize_telegram_chat_scope(json_value(body, "chat_scope", std::string()));
        const std::string decision = trim_ascii_copy(json_value(body, "decision", std::string()));
        const std::string selected_option_label = trim_ascii_copy(json_value(body, "selected_option_label", std::string()));
        const int32_t selected_option_index = std::max<int32_t>(-1, json_value(body, "selected_option_index", -1));

        if (approval_id.empty()) {
            return {
                {"ok", false},
                {"status", "expired"},
                {"error", "approval_id is required"},
            };
        }

        vicuna_pending_approval approval = {};
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            auto it = pending_approvals_by_id.find(approval_id);
            if (it == pending_approvals_by_id.end()) {
                return {
                    {"ok", false},
                    {"approval_id", approval_id},
                    {"status", "expired"},
                    {"error", "approval prompt is no longer active"},
                };
            }

            approval = it->second;
            if (!chat_scope.empty() && approval.chat_scope != chat_scope) {
                return {
                    {"ok", false},
                    {"approval_id", approval_id},
                    {"status", pending_approval_status_name(approval.status)},
                    {"error", "approval chat scope did not match"},
                };
            }
            if (approval.status != VICUNA_PENDING_APPROVAL_PENDING_PUBLISH &&
                    approval.status != VICUNA_PENDING_APPROVAL_AWAITING_USER) {
                return {
                    {"ok", false},
                    {"approval_id", approval_id},
                    {"status", pending_approval_status_name(approval.status)},
                    {"error", "approval prompt is no longer active"},
                };
            }
            if (selected_option_index < 0 ||
                    selected_option_index >= (int32_t) approval.options.size()) {
                return {
                    {"ok", false},
                    {"approval_id", approval_id},
                    {"status", pending_approval_status_name(approval.status)},
                    {"error", "approval option index was invalid"},
                };
            }
            if (!selected_option_label.empty() &&
                    approval.options[(size_t) selected_option_index] != selected_option_label) {
                return {
                    {"ok", false},
                    {"approval_id", approval_id},
                    {"status", pending_approval_status_name(approval.status)},
                    {"error", "approval option label did not match the indexed option"},
                };
            }

            const std::string resolved_option = approval.options[(size_t) selected_option_index];
            const bool disallowed =
                    trim_ascii_copy(decision) == "disallow" ||
                    trim_ascii_copy(resolved_option) == "disallow";
            it->second.status = disallowed ? VICUNA_PENDING_APPROVAL_DENIED : VICUNA_PENDING_APPROVAL_APPROVED;
            it->second.updated_at_ms = ggml_time_ms();
            it->second.resolved_at_ms = it->second.updated_at_ms;
            it->second.resolution_detail = disallowed ?
                    "operator disallowed command before execution" :
                    "operator approved command execution";
            approval = it->second;
        }

        mark_runtime_state_dirty("approval-decision");
        capture_tool_result_provenance("approval_decision", pending_approval_to_json(approval), nullptr);
        server_task wake_task(SERVER_TASK_TYPE_NEXT_RESPONSE);
        wake_task.id = queue_tasks.get_new_id();
        queue_tasks.post(std::move(wake_task), true);
        return {
            {"ok", true},
            {"approval_id", approval.approval_id},
            {"status", pending_approval_status_name(approval.status)},
            {"command_id", approval.command_id},
            {"origin", approval.origin == LLAMA_COG_COMMAND_ORIGIN_DMN ? "dmn" : "active"},
            {"dispatched", false},
        };
    }

    json interrupt_pending_dmn_approvals_for_chat(const json & body) {
        const std::string chat_scope = normalize_telegram_chat_scope(json_value(body, "chat_scope", std::string()));
        const int64_t telegram_message_id = std::max<int64_t>(0, json_value(body, "telegram_message_id", int64_t(0)));
        json cancelled = json::array();
        if (chat_scope.empty()) {
            return {
                {"ok", false},
                {"error", "chat_scope is required"},
                {"cancelled_dmn_approval_ids", cancelled},
            };
        }

        bool changed = false;
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            for (auto & [approval_id, approval] : pending_approvals_by_id) {
                if (approval.origin != LLAMA_COG_COMMAND_ORIGIN_DMN ||
                        approval.chat_scope != chat_scope) {
                    continue;
                }
                if (approval.status != VICUNA_PENDING_APPROVAL_PENDING_PUBLISH &&
                        approval.status != VICUNA_PENDING_APPROVAL_AWAITING_USER) {
                    continue;
                }
                approval.status = VICUNA_PENDING_APPROVAL_SUPERSEDED;
                approval.updated_at_ms = ggml_time_ms();
                approval.resolved_at_ms = approval.updated_at_ms;
                approval.resolution_detail =
                        telegram_message_id > 0 ?
                                "superseded by a newer Telegram user message" :
                                "superseded by new Telegram input";
                cancelled.push_back(approval_id);
                changed = true;
            }
        }

        if (changed) {
            mark_runtime_state_dirty("dmn-approval-interrupt");
            server_task wake_task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            wake_task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(wake_task), true);
        }
        return {
            {"ok", true},
            {"cancelled_dmn_approval_ids", cancelled},
        };
    }

    json build_telegram_outbox_health_json() const {
        const vicuna_telegram_outbox_snapshot snapshot = telegram_outbox_snapshot_copy(telegram_outbox);
        uint64_t oldest_sequence = 0;
        uint64_t newest_sequence = 0;
        if (!snapshot.items.empty()) {
            oldest_sequence = snapshot.items.front().sequence_number;
            newest_sequence = snapshot.items.back().sequence_number;
        }
        return {
            {"stored_items", (int) snapshot.items.size()},
            {"publish_total", snapshot.publish_total},
            {"dropped_total", snapshot.dropped_total},
            {"last_publish_ms", snapshot.last_publish_ms},
            {"next_sequence_number", snapshot.next_sequence_number},
            {"oldest_sequence", oldest_sequence},
            {"newest_sequence", newest_sequence},
        };
    }

    json build_pending_approvals_health_json() const {
        json approvals = json::array();
        size_t total_count = 0;
        size_t pending_publish = 0;
        size_t awaiting_user = 0;
        size_t approved_waiting_dispatch = 0;
        size_t active_origin = 0;
        size_t dmn_origin = 0;

        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        total_count = pending_approvals_by_id.size();
        for (const auto & [approval_id, approval] : pending_approvals_by_id) {
            GGML_UNUSED(approval_id);
            approvals.push_back(pending_approval_to_json(approval));
            if (approval.status == VICUNA_PENDING_APPROVAL_PENDING_PUBLISH) {
                ++pending_publish;
            } else if (approval.status == VICUNA_PENDING_APPROVAL_AWAITING_USER) {
                ++awaiting_user;
            } else if (approval.status == VICUNA_PENDING_APPROVAL_APPROVED) {
                ++approved_waiting_dispatch;
            }
            if (approval.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
                ++dmn_origin;
            } else if (approval.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
                ++active_origin;
            }
        }

        return {
            {"count", total_count},
            {"pending_publish", pending_publish},
            {"awaiting_user", awaiting_user},
            {"approved_waiting_dispatch", approved_waiting_dispatch},
            {"active_origin", active_origin},
            {"dmn_origin", dmn_origin},
            {"items", approvals},
        };
    }

    json build_foreground_consent_health_json() const {
        json items = json::array();
        std::lock_guard<std::mutex> lock(runtime_state_mutex);
        for (const auto & [episode_id, scope] : foreground_consent_by_episode) {
            GGML_UNUSED(episode_id);
            items.push_back({
                {"scope_id", scope.scope_id},
                {"task_id", scope.task_id},
                {"episode_id", scope.episode_id},
                {"chat_scope", scope.chat_scope},
                {"telegram_message_id", scope.telegram_message_id},
                {"created_at_ms", scope.created_at_ms},
                {"updated_at_ms", scope.updated_at_ms},
            });
        }
        return {
            {"count", items.size()},
            {"items", std::move(items)},
        };
    }

    void telegram_dialogue_prune_scope_locked(const std::string & scope) const {
        auto it = telegram_dialogue_history.turns_by_scope.find(scope);
        if (it == telegram_dialogue_history.turns_by_scope.end()) {
            return;
        }
        while (it->second.size() > telegram_dialogue_history.max_turns_per_scope) {
            it->second.pop_front();
        }
        if (it->second.empty()) {
            telegram_dialogue_history.turns_by_scope.erase(it);
            telegram_dialogue_history.latest_message_id_by_scope.erase(scope);
        }
    }

    std::string dmn_telegram_dialogue_scope_locked() const {
        std::string selected_scope;
        int scope_count = 0;
        for (const auto & [scope, turns] : telegram_dialogue_history.turns_by_scope) {
            if (turns.empty() || telegram_dialogue_scope_is_broadcast(scope)) {
                continue;
            }
            selected_scope = scope;
            ++scope_count;
            if (scope_count > 1) {
                return std::string();
            }
        }
        return scope_count == 1 ? selected_scope : std::string();
    }

    void sync_telegram_dialogue_history(
            const std::string & chat_scope,
            int64_t telegram_message_id,
            const std::vector<common_chat_msg> & transcript_messages) const {
        const std::string normalized_scope = normalize_telegram_chat_scope(chat_scope);
        if (normalized_scope.empty() || transcript_messages.empty()) {
            return;
        }

        const int64_t now_ms = ggml_time_ms();
        std::lock_guard<std::mutex> lock(telegram_dialogue_history.mutex);
        std::deque<vicuna_telegram_dialogue_turn> rebuilt_turns;
        rebuilt_turns.clear();

        auto append_turn = [&](const common_chat_msg & msg) {
            const std::string role = trim_ascii_copy(msg.role);
            const std::string content = trim_ascii_copy(msg.content);
            if (content.empty()) {
                return;
            }

            if (role == "user") {
                if (!rebuilt_turns.empty() &&
                    rebuilt_turns.back().assistant_text.empty() &&
                    !rebuilt_turns.back().user_text.empty()) {
                    rebuilt_turns.back().user_text = content;
                    rebuilt_turns.back().updated_at_ms = now_ms;
                } else {
                    vicuna_telegram_dialogue_turn turn = {};
                    turn.turn_id = telegram_dialogue_history.next_turn_id++;
                    turn.chat_scope = normalized_scope;
                    turn.user_text = content;
                    turn.updated_at_ms = now_ms;
                    rebuilt_turns.push_back(std::move(turn));
                }
            } else if (role == "assistant") {
                if (!rebuilt_turns.empty() &&
                    rebuilt_turns.back().assistant_text.empty() &&
                    !rebuilt_turns.back().user_text.empty()) {
                    rebuilt_turns.back().assistant_text = content;
                    rebuilt_turns.back().assistant_source = "telegram_bridge_request";
                    rebuilt_turns.back().updated_at_ms = now_ms;
                } else {
                    vicuna_telegram_dialogue_turn turn = {};
                    turn.turn_id = telegram_dialogue_history.next_turn_id++;
                    turn.chat_scope = normalized_scope;
                    turn.assistant_text = content;
                    turn.assistant_source = "telegram_bridge_request";
                    turn.updated_at_ms = now_ms;
                    rebuilt_turns.push_back(std::move(turn));
                }
            }
        };

        for (const auto & msg : transcript_messages) {
            append_turn(msg);
        }

        while (rebuilt_turns.size() > telegram_dialogue_history.max_turns_per_scope) {
            rebuilt_turns.pop_front();
        }

        if (!rebuilt_turns.empty()) {
            telegram_dialogue_history.turns_by_scope[normalized_scope] = std::move(rebuilt_turns);
            if (telegram_message_id > 0) {
                telegram_dialogue_history.latest_message_id_by_scope[normalized_scope] = std::max(
                        telegram_dialogue_history.latest_message_id_by_scope[normalized_scope],
                        telegram_message_id);
            }
        }
    }

    void append_telegram_dialogue_assistant_turn(
            const std::string & chat_scope,
            const std::string & text,
            const char * source,
            const std::string & response_id = std::string(),
            const std::string & dedupe_key = std::string()) const {
        const std::string normalized_scope = normalize_telegram_chat_scope(chat_scope);
        const std::string trimmed = trim_ascii_copy(text);
        if (normalized_scope.empty() || trimmed.empty()) {
            return;
        }

        const int64_t now_ms = ggml_time_ms();
        std::lock_guard<std::mutex> lock(telegram_dialogue_history.mutex);
        std::deque<vicuna_telegram_dialogue_turn> & turns =
                telegram_dialogue_history.turns_by_scope[normalized_scope];

        if (!turns.empty() &&
            turns.back().assistant_text.empty() &&
            !turns.back().user_text.empty()) {
            turns.back().assistant_text = trimmed;
            turns.back().assistant_source = source ? source : "";
            turns.back().response_id = response_id;
            turns.back().dedupe_key = dedupe_key;
            turns.back().updated_at_ms = now_ms;
        } else {
            vicuna_telegram_dialogue_turn turn = {};
            turn.turn_id = telegram_dialogue_history.next_turn_id++;
            turn.chat_scope = normalized_scope;
            turn.assistant_text = trimmed;
            turn.assistant_source = source ? source : "";
            turn.response_id = response_id;
            turn.dedupe_key = dedupe_key;
            turn.updated_at_ms = now_ms;
            turns.push_back(std::move(turn));
        }

        telegram_dialogue_prune_scope_locked(normalized_scope);
    }

    bool task_wants_deferred_telegram_delivery(const server_task & task) const {
        return task.react_origin == SERVER_REACT_ORIGIN_ACTIVE &&
               task.telegram_dialogue_active &&
               task.telegram_deferred_delivery &&
               !normalize_telegram_chat_scope(task.telegram_chat_scope).empty();
    }

    std::string deferred_telegram_delivery_dedupe_key(
            const server_task & task,
            const char * suffix) const {
        std::ostringstream key;
        key << "active-telegram-"
            << task.id << "-"
            << (task.has_active_trace ? task.active_trace.episode_id : -1);
        if (suffix && suffix[0] != '\0') {
            key << "-" << suffix;
        }
        return key.str();
    }

    bool publish_deferred_telegram_delivery(
            const server_task & task,
            const std::string & text,
            const char * source,
            const char * dedupe_suffix,
            bool reply_to_origin = true) const {
        if (!task_wants_deferred_telegram_delivery(task)) {
            return false;
        }

        const std::string trimmed = trim_ascii_copy(text);
        if (trimmed.empty()) {
            return false;
        }

        const std::string dedupe_key = deferred_telegram_delivery_dedupe_key(task, dedupe_suffix);
        if (!telegram_outbox_publish_message(
                    task.telegram_chat_scope,
                    dedupe_key,
                    trimmed,
                    reply_to_origin ? task.telegram_message_id : 0)) {
            return false;
        }

        append_telegram_dialogue_assistant_turn(
                task.telegram_chat_scope,
                trimmed,
                source,
                std::string(),
                dedupe_key);
        mark_runtime_state_dirty("telegram-outbox-active-message");
        return true;
    }

    bool publish_terminal_deferred_telegram_delivery(
            const server_task & task,
            const std::string & text,
            const char * source,
            const char * dedupe_suffix) const {
        if (!task_wants_deferred_telegram_delivery(task)) {
            return false;
        }

        const std::string trimmed = trim_ascii_copy(text);
        if (!trimmed.empty()) {
            return publish_deferred_telegram_delivery(task, trimmed, source, dedupe_suffix);
        }

        const std::string fallback = synthesize_deferred_terminal_failure_text(task.foreground_text);
        const bool published = publish_deferred_telegram_delivery(
                task,
                fallback,
                "telegram_relay_active_terminal_fallback",
                "terminal-fallback");
        if (published) {
            SRV_WRN("synthesized deferred terminal follow-up for task %d after empty terminal assistant text\n",
                    task.id);
        }
        return published;
    }

    std::vector<common_chat_msg> telegram_dialogue_messages_for_task(const server_task & task) const {
        const bool active_scope = task.telegram_dialogue_active &&
                                  !normalize_telegram_chat_scope(task.telegram_chat_scope).empty();
        const bool dmn_scope = task.react_origin == SERVER_REACT_ORIGIN_DMN;
        if (!active_scope && !dmn_scope) {
            return {};
        }

        const int32_t turn_limit = std::max(
                1,
                task.telegram_history_turn_limit > 0 ?
                        task.telegram_history_turn_limit :
                        VICUNA_TELEGRAM_DIALOGUE_DEFAULT_TURN_LIMIT);

        std::vector<vicuna_telegram_dialogue_turn> selected_turns;
        {
            std::lock_guard<std::mutex> lock(telegram_dialogue_history.mutex);
            const std::string scope =
                    active_scope ?
                            normalize_telegram_chat_scope(task.telegram_chat_scope) :
                            dmn_telegram_dialogue_scope_locked();

            auto append_scope_turns = [&](const std::string & target_scope) {
                const auto it = telegram_dialogue_history.turns_by_scope.find(target_scope);
                if (it == telegram_dialogue_history.turns_by_scope.end()) {
                    return;
                }
                selected_turns.insert(selected_turns.end(), it->second.begin(), it->second.end());
            };

            append_scope_turns(VICUNA_TELEGRAM_DIALOGUE_SCOPE_BROADCAST);
            if (!scope.empty() && !telegram_dialogue_scope_is_broadcast(scope)) {
                append_scope_turns(scope);
            }
        }

        if (selected_turns.empty()) {
            return {};
        }

        std::sort(
                selected_turns.begin(),
                selected_turns.end(),
                [](const vicuna_telegram_dialogue_turn & lhs, const vicuna_telegram_dialogue_turn & rhs) {
                    if (lhs.updated_at_ms != rhs.updated_at_ms) {
                        return lhs.updated_at_ms < rhs.updated_at_ms;
                    }
                    return lhs.turn_id < rhs.turn_id;
                });

        if ((int32_t) selected_turns.size() > turn_limit) {
            selected_turns.erase(selected_turns.begin(), selected_turns.end() - turn_limit);
        }

        std::vector<common_chat_msg> messages;
        for (const auto & turn : selected_turns) {
            if (!turn.user_text.empty()) {
                messages.push_back(common_chat_msg{
                        /*.role =*/ "user",
                        /*.content =*/ turn.user_text,
                        /*.content_parts =*/ {},
                        /*.tool_calls =*/ {},
                        /*.reasoning_content =*/ {},
                        /*.tool_name =*/ {},
                        /*.tool_call_id =*/ {},
                });
            }
            if (!turn.assistant_text.empty()) {
                messages.push_back(common_chat_msg{
                        /*.role =*/ "assistant",
                        /*.content =*/ turn.assistant_text,
                        /*.content_parts =*/ {},
                        /*.tool_calls =*/ {},
                        /*.reasoning_content =*/ {},
                        /*.tool_name =*/ {},
                        /*.tool_call_id =*/ {},
                });
            }
        }
        return messages;
    }

    json build_telegram_dialogue_health_json() const {
        const vicuna_telegram_dialogue_snapshot snapshot = telegram_dialogue_snapshot_copy(telegram_dialogue_history);
        size_t total_turns = 0;
        for (const auto & [scope, turns] : snapshot.turns_by_scope) {
            GGML_UNUSED(scope);
            total_turns += turns.size();
        }
        return {
            {"scope_count", (int) snapshot.turns_by_scope.size()},
            {"turn_count", (int) total_turns},
            {"max_turns_per_scope", (int) snapshot.max_turns_per_scope},
        };
    }

    bool dispatch_pending_self_emit_commands(int32_t origin_filter) {
        if (!ctx) {
            return false;
        }

        bool published = false;
        const int32_t command_count = llama_cognitive_command_count(ctx);
        for (int32_t i = 0; i < command_count; ++i) {
            llama_cognitive_command command = {};
            if (llama_cognitive_command_get(ctx, i, &command) != 0) {
                continue;
            }
            if (command.kind != LLAMA_COG_COMMAND_EMIT_BACKGROUND ||
                command.status != LLAMA_COG_COMMAND_STATUS_PENDING) {
                continue;
            }
            if (origin_filter >= 0 && command.origin != origin_filter) {
                continue;
            }
            if (llama_cognitive_command_ack(ctx, command.command_id) != 0) {
                SRV_WRN("failed to ack proactive emit command %d\n", command.command_id);
                continue;
            }

            llama_dmn_tick_trace dmn_trace = {};
            llama_remediation_plan remediation = {};
            llama_governance_trace governance = {};
            llama_self_model_state_info model_state = {};
            (void) llama_dmn_get_last_trace(ctx, &dmn_trace);
            (void) llama_remediation_get_last_plan(ctx, &remediation);
            (void) llama_governance_get_last_trace(ctx, &governance);
            (void) llama_self_state_get_model_state(ctx, &model_state);

            float directness_preference = 0.5f;
            float verbosity_preference = 0.5f;
            if (model_state.horizon_count > 0) {
                const auto & preference = model_state.horizons[(size_t) LLAMA_SELF_HORIZON_INSTANT].user_preference;
                directness_preference = preference.directness_preference;
                verbosity_preference = preference.verbosity_preference;
            }

            const std::string message = proactive_mailbox_message(
                    dmn_trace,
                    remediation,
                    governance,
                    directness_preference,
                    verbosity_preference);
            std::string response_id;
            if (!proactive_mailbox_publish(message, "dmn", &response_id)) {
                {
                    std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
                    proactive_mailbox.fail_total++;
                }
                SRV_WRN("proactive emit command %d produced an empty message\n", command.command_id);
            } else {
                append_telegram_dialogue_assistant_turn(
                        VICUNA_TELEGRAM_DIALOGUE_SCOPE_BROADCAST,
                        message,
                        "telegram_relay_dmn",
                        response_id,
                        std::string("dmn-self-emit-") + std::to_string(command.command_id));
                if (!admit_runtime_emit_text(
                            ctx,
                            message,
                            command.origin,
                            LLAMA_FUNCTIONAL_MICROPHASE_NONE,
                            command.command_id,
                            -1,
                            LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP)) {
                    SRV_WRN("failed to admit proactive emit command %d into self-state\n", command.command_id);
                }
                published = true;
                mark_runtime_state_dirty("proactive-self-emit");
            }

            if (llama_cognitive_command_complete(ctx, command.command_id, false) != 0) {
                SRV_WRN("failed to complete proactive emit command %d\n", command.command_id);
            }
        }

        return published;
    }

    void inject_startup_self_emit_from_env() {
        const char * startup_text = std::getenv("VICUNA_SELF_EMIT_STARTUP_TEXT");
        if (!startup_text || startup_text[0] == '\0') {
            return;
        }
        if (proactive_mailbox_publish(startup_text, "startup")) {
            if (!admit_runtime_emit_text(
                        ctx,
                        startup_text,
                        LLAMA_COG_COMMAND_ORIGIN_DMN,
                        LLAMA_FUNCTIONAL_MICROPHASE_NONE,
                        -1,
                        -1,
                        LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP)) {
                SRV_WRN("%s\n", "failed to admit startup self-emit into self-state");
            }
            mark_runtime_state_dirty("startup-self-emit");
        }
    }

    void destroy() {
        if (ctx) {
            (void) persist_runtime_state("destroy");
        }
        proactive_mailbox_reset_live_stream();
        stop_external_workers();
        llama_init.reset();
        ctx = nullptr;
        model = nullptr;
        state = SERVER_STATE_LOADING_MODEL;

        mtmd_free(mctx);
        mctx = nullptr;

        // Clear any sampling context
        for (server_slot & slot : slots) {
            common_speculative_free(slot.spec);
            slot.spec = nullptr;
        }

        llama_batch_free(batch);
    }

    bool execute_codex_tool_request(
            const llama_codex_tool_request & request,
            llama_codex_tool_result * out_result) {
        if (!out_result) {
            return false;
        }

        *out_result = {};
        out_result->command_id = request.command_id;
        out_result->tool_job_id = request.tool_job_id;

        const std::string repo_root = trim_ascii_copy(request.working_directory);
        const std::string codex_path = trim_ascii_copy(request.codex_path);
        const std::string task_prompt = trim_ascii_copy(request.task_prompt);
        if (repo_root.empty() || codex_path.empty() || task_prompt.empty() || !request.command_ready) {
            out_result->launch_failed = true;
            out_result->exit_code = 127;
            std::snprintf(out_result->error_text, sizeof(out_result->error_text), "%s", "codex tool request was incomplete");
            std::snprintf(out_result->summary_text, sizeof(out_result->summary_text), "%s", "Codex tool request was incomplete.");
            return true;
        }

        const std::filesystem::path temp_root =
                std::filesystem::temp_directory_path() /
                std::filesystem::path("vicuna-codex-" + std::to_string((long long) request.tool_job_id) + "-" + random_string());
        const std::filesystem::path prompt_path = temp_root / "prompt.txt";
        const std::filesystem::path summary_path = temp_root / "last-message.txt";

        std::string write_error;
        if (!write_text_file(prompt_path, task_prompt, &write_error)) {
            out_result->launch_failed = true;
            out_result->exit_code = 127;
            std::snprintf(out_result->error_text, sizeof(out_result->error_text), "%s", write_error.c_str());
            std::snprintf(out_result->summary_text, sizeof(out_result->summary_text), "%s", "Failed to prepare Codex prompt file.");
            return true;
        }

        const std::string codex_command =
                shell_quote(codex_path) +
                " exec --dangerously-bypass-approvals-and-sandbox --sandbox danger-full-access" +
                " -C " + shell_quote(repo_root) +
                " -o " + shell_quote(summary_path.string()) +
                " - < " + shell_quote(prompt_path.string());

        llama_bash_tool_result exec_result = {};
        (void) run_permissive_bash_command(
                repo_root,
                codex_command,
                request.timeout_ms,
                std::min(request.max_stdout_bytes, LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1),
                std::min(request.max_stderr_bytes, LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1),
                &exec_result);

        out_result->launch_failed = exec_result.launch_failed;
        out_result->exit_code = exec_result.exit_code;
        out_result->runtime_ms = exec_result.runtime_ms;
        out_result->truncated_stdout = exec_result.truncated_stdout;
        out_result->truncated_stderr = exec_result.truncated_stderr;
        std::snprintf(out_result->stdout_text, sizeof(out_result->stdout_text), "%s", exec_result.stdout_text);
        std::snprintf(out_result->stderr_text, sizeof(out_result->stderr_text), "%s", exec_result.stderr_text);
        std::snprintf(out_result->error_text, sizeof(out_result->error_text), "%s", exec_result.error_text);

        const std::string last_message = read_text_file_bounded(summary_path, LLAMA_CODEX_TOOL_SUMMARY_MAX_CHARS - 1);
        const std::string manual_requirements = parse_manual_requirements_line(last_message);
        const std::string summary_text = remove_manual_requirements_line(last_message);
        std::snprintf(out_result->summary_text, sizeof(out_result->summary_text), "%s",
                (!summary_text.empty() ? summary_text : "Codex run completed.").c_str());
        if (!manual_requirements.empty() &&
                string_to_lower(manual_requirements) != "none") {
            std::snprintf(out_result->manual_requirements, sizeof(out_result->manual_requirements), "%s", manual_requirements.c_str());
        }

        llama_bash_tool_result status_result = {};
        (void) run_permissive_bash_command(
                repo_root,
                "git status --short",
                10000,
                LLAMA_BASH_TOOL_STDOUT_MAX_CHARS - 1,
                LLAMA_BASH_TOOL_STDERR_MAX_CHARS - 1,
                &status_result);
        const std::string status_text = trim_ascii_copy(status_result.stdout_text);
        out_result->repo_changed = !status_text.empty();
        if (!status_text.empty()) {
            std::snprintf(out_result->changed_files_excerpt, sizeof(out_result->changed_files_excerpt), "%s", status_text.c_str());
        }

        std::error_code cleanup_ec;
        std::filesystem::remove(summary_path, cleanup_ec);
        std::filesystem::remove(prompt_path, cleanup_ec);
        std::filesystem::remove(temp_root, cleanup_ec);
        return true;
    }

    static void run_external_worker(
            vicuna_external_work_queue * queue,
            vicuna_external_work_kind kind,
            server_queue * task_queue,
            server_context_impl * self) {
        GGML_ASSERT(queue != nullptr);
        GGML_ASSERT(task_queue != nullptr);
        GGML_ASSERT(self != nullptr);
        while (true) {
            vicuna_external_work_item work = {};
            {
                std::unique_lock<std::mutex> lock(queue->mutex);
                queue->cv.wait(lock, [&]() {
                    return queue->stop || !queue->pending.empty();
                });
                if (queue->stop && queue->pending.empty()) {
                    return;
                }
                work = queue->pending.front();
                queue->pending.pop_front();
            }

            vicuna_external_work_result result = {};
            result.command_id = work.command_id;
            result.origin = work.origin;
            result.tool_kind = work.tool_kind;
            result.kind = kind;

            if (kind == VICUNA_EXTERNAL_WORK_BASH) {
                (void) server_bash_tool_execute(work.bash_request, &result.bash_result);
            } else if (kind == VICUNA_EXTERNAL_WORK_HARD_MEMORY) {
                llama_hard_memory helper;
                (void) helper.configure(work.hard_memory_config);
                result.hard_memory_result.command_id = work.command_id;
                result.hard_memory_result.tool_job_id = work.hard_memory_request.tool_job_id;
                result.hard_memory_result.operation = work.hard_memory_request.operation;
                if (work.hard_memory_request.operation == LLAMA_COG_HARD_MEMORY_OPERATION_WRITE) {
                    (void) helper.archive_write_items(
                            work.hard_memory_request.write_items,
                            work.hard_memory_request.write_count,
                            work.hard_memory_request.container_tag[0] != '\0' ? work.hard_memory_request.container_tag : nullptr,
                            nullptr);
                    (void) helper.get_last_archive_trace(&result.hard_memory_result.archive_trace);
                } else {
                    (void) helper.query(work.hard_memory_request.query, &result.hard_memory_result.result);
                }
            } else {
                result.codex_request = work.codex_request;
                (void) self->execute_codex_tool_request(work.codex_request, &result.codex_result);
            }
            result.completed_ms = ggml_time_ms();

            {
                std::lock_guard<std::mutex> lock(queue->mutex);
                queue->inflight_ids.erase(work.command_id);
                queue->completed.push_back(std::move(result));
            }
            server_task wake_task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            wake_task.id = task_queue->get_new_id();
            task_queue->post(std::move(wake_task), true);
        }
    }

    void start_external_workers() {
        stop_external_workers();
        {
            std::lock_guard<std::mutex> lock(bash_work.mutex);
            bash_work.stop = false;
        }
        {
            std::lock_guard<std::mutex> lock(hard_memory_work.mutex);
            hard_memory_work.stop = false;
        }
        {
            std::lock_guard<std::mutex> lock(codex_work.mutex);
            codex_work.stop = false;
        }
        bash_worker = std::thread(&server_context_impl::run_external_worker, &bash_work, VICUNA_EXTERNAL_WORK_BASH, &queue_tasks, this);
        hard_memory_worker = std::thread(&server_context_impl::run_external_worker, &hard_memory_work, VICUNA_EXTERNAL_WORK_HARD_MEMORY, &queue_tasks, this);
        codex_worker = std::thread(&server_context_impl::run_external_worker, &codex_work, VICUNA_EXTERNAL_WORK_CODEX, &queue_tasks, this);
    }

    void stop_external_workers() {
        {
            std::lock_guard<std::mutex> lock(bash_work.mutex);
            bash_work.stop = true;
            bash_work.pending.clear();
            bash_work.completed.clear();
            bash_work.inflight_ids.clear();
        }
        bash_work.cv.notify_all();
        if (bash_worker.joinable()) {
            bash_worker.join();
        }

        {
            std::lock_guard<std::mutex> lock(hard_memory_work.mutex);
            hard_memory_work.stop = true;
            hard_memory_work.pending.clear();
            hard_memory_work.completed.clear();
            hard_memory_work.inflight_ids.clear();
        }
        hard_memory_work.cv.notify_all();
        if (hard_memory_worker.joinable()) {
            hard_memory_worker.join();
        }
        {
            std::lock_guard<std::mutex> lock(codex_work.mutex);
            codex_work.stop = true;
            codex_work.pending.clear();
            codex_work.completed.clear();
            codex_work.inflight_ids.clear();
        }
        codex_work.cv.notify_all();
        if (codex_worker.joinable()) {
            codex_worker.join();
        }
    }

    void reload_openclaw_catalog_if_needed() {
        if (!ctx || !openclaw_fabric.enabled()) {
            return;
        }

        bool reloaded = false;
        std::string reload_error;
        if (!openclaw_fabric.maybe_reload(
                    bash_tool_enabled,
                    hard_memory_enabled,
                    codex_tool_enabled,
                    &reloaded,
                    &reload_error)) {
            SRV_WRN("failed to reload OpenClaw tool fabric: %s\n", reload_error.c_str());
            return;
        }
        if (!reloaded) {
            return;
        }

        std::vector<llama_cognitive_tool_spec> specs;
        if (openclaw_fabric.build_cognitive_specs(&specs) &&
                !specs.empty() &&
                llama_cognitive_tool_spec_set(ctx, specs.data(), (int32_t) specs.size()) == 0) {
            SRV_INF("OpenClaw tool fabric reloaded with %zu capabilities\n", specs.size());
        } else {
            SRV_WRN("%s\n", "failed to install reloaded OpenClaw tool catalog into cognitive runtime");
        }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    bool persist_runtime_state(const char * reason) {
        std::string snapshot_path;
        std::vector<vicuna_pending_approval> pending_approvals;
        uint64_t pending_approval_ordinal = 1;
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            if (!runtime_persistence.enabled || runtime_persistence.snapshot_path.empty() || !runtime_state_dirty) {
                return true;
            }
            snapshot_path = runtime_persistence.snapshot_path;
            pending_approvals.reserve(pending_approvals_by_id.size());
            for (const auto & [approval_id, approval] : pending_approvals_by_id) {
                GGML_UNUSED(approval_id);
                pending_approvals.push_back(approval);
            }
            pending_approval_ordinal = next_pending_approval_ordinal;
        }

        if (!ctx) {
            return false;
        }

        try {
            json snapshot = json::object();
            snapshot["version"] = 8;
            snapshot["saved_at_unix_ms"] = ggml_time_ms();
            snapshot["reason"] = reason ? reason : "";

            llama_bash_tool_config bash_config = llama_bash_tool_default_config();
            llama_hard_memory_config hard_memory_config = llama_hard_memory_default_config();
            llama_self_updater_program updater = llama_self_state_default_updater_program();
            const size_t trace_size = llama_self_state_trace_export_size(ctx);
            std::vector<uint8_t> trace_blob(trace_size);

            if (llama_bash_tool_get_config(ctx, &bash_config) != 0 ||
                llama_hard_memory_get_config(ctx, &hard_memory_config) != 0 ||
                llama_self_state_get_updater_program(ctx, &updater) != 0 ||
                (trace_size > 0 && llama_self_state_trace_export(ctx, trace_blob.data(), trace_blob.size()) != 0)) {
                throw std::runtime_error("failed to gather runtime state for persistence");
            }

            snapshot["bash_tool_config"] = bash_tool_config_to_json(bash_config);
            snapshot["hard_memory_config"] = hard_memory_config_to_json(hard_memory_config);
            snapshot["self_state_updater_program_b64"] = base64::encode((const char *) &updater, sizeof(updater));
            snapshot["self_state_trace_b64"] = base64::encode((const char *) trace_blob.data(), trace_blob.size());
            snapshot["proactive_mailbox"] = proactive_mailbox_to_json(proactive_mailbox);
            snapshot["telegram_outbox"] = telegram_outbox_to_json(telegram_outbox);
            snapshot["telegram_dialogue"] = telegram_dialogue_history_to_json(telegram_dialogue_history);
            json approval_items = json::array();
            for (const auto & approval : pending_approvals) {
                approval_items.push_back(pending_approval_to_json(approval));
            }
            snapshot["pending_approvals"] = {
                {"next_ordinal", pending_approval_ordinal},
                {"items", std::move(approval_items)},
            };

            json extensions = json::array();
            const int32_t extension_count = llama_self_state_model_extension_count(ctx);
            for (int32_t i = 0; i < extension_count; ++i) {
                llama_self_model_extension_info info = {};
                if (llama_self_state_get_model_extension(ctx, i, &info) == 0) {
                    extensions.push_back(model_extension_info_to_json(info));
                }
            }
            snapshot["model_extensions"] = std::move(extensions);

            llama_functional_lora_family_state functional_state = {};
            if (llama_functional_lora_family_state_get(ctx, 0, &functional_state) == 0) {
                json functional_snapshots = json::array();
                for (int32_t family = 0; family < LLAMA_FUNCTIONAL_LORA_COUNT; ++family) {
                    llama_functional_lora_snapshot_archive archive = {};
                    if (llama_functional_lora_snapshot_archive_get(ctx, family, &archive) != 0) {
                        throw std::runtime_error("failed to query functional snapshot archive");
                    }
                    json family_entry = json::object();
                    family_entry["family"] = family;
                    family_entry["count"] = archive.count;
                    family_entry["last_capture_us"] = archive.last_capture_us;
                    family_entry["next_capture_due_us"] = archive.next_capture_due_us;
                    json items = json::array();
                    for (int32_t slot = 0; slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++slot) {
                        llama_functional_lora_snapshot_info info = {};
                        if (llama_functional_lora_snapshot_info_get(ctx, family, slot, &info) != 0) {
                            throw std::runtime_error("failed to query functional snapshot info");
                        }
                        if (!info.valid) {
                            continue;
                        }
                        const size_t blob_size = llama_functional_lora_snapshot_blob_size(ctx, family, slot);
                        if (blob_size == 0) {
                            throw std::runtime_error("functional snapshot archive reported empty blob");
                        }
                        std::vector<uint8_t> blob(blob_size);
                        if (llama_functional_lora_snapshot_blob_export(ctx, family, slot, blob.data(), blob.size()) != 0) {
                            throw std::runtime_error("failed to export functional snapshot blob");
                        }

                        json item = json::object();
                        item["valid"] = info.valid;
                        item["family"] = info.family;
                        item["slot"] = info.slot;
                        item["source"] = info.source;
                        item["snapshot_id"] = info.snapshot_id;
                        item["captured_at_us"] = info.captured_at_us;
                        item["expires_at_us"] = info.expires_at_us;
                        item["source_update_count"] = info.source_update_count;
                        item["self_state_gradient_norm"] = info.self_state_gradient_norm;
                        item["robustness_score"] = info.robustness_score;
                        item["last_signed_outcome"] = info.last_signed_outcome;
                        item["dominant_direction_cosine"] = info.dominant_direction_cosine;
                        item["blob_b64"] = base64::encode((const char *) blob.data(), blob.size());
                        items.push_back(std::move(item));
                    }
                    family_entry["items"] = std::move(items);
                    functional_snapshots.push_back(std::move(family_entry));
                }
                snapshot["functional_snapshots"] = std::move(functional_snapshots);
            }

            llama_process_functional_params process_params = llama_process_functional_default_params();
            if (llama_process_functional_get_params(ctx, &process_params) == 0) {
                json process_params_json = json::object();
                process_params_json["enabled"] = process_params.enabled;
                process_params_json["max_entries"] = process_params.max_entries;
                process_params_json["min_observations"] = process_params.min_observations;
                process_params_json["noop_abs_ceiling"] = process_params.noop_abs_ceiling;
                process_params_json["weak_positive_ceiling"] = process_params.weak_positive_ceiling;
                process_params_json["mean_outcome_ceiling"] = process_params.mean_outcome_ceiling;
                process_params_json["weak_or_worse_ratio_threshold"] = process_params.weak_or_worse_ratio_threshold;
                process_params_json["creation_cooldown_updates"] = process_params.creation_cooldown_updates;
                process_params_json["utility_decay"] = process_params.utility_decay;
                snapshot["process_functional_params"] = std::move(process_params_json);

                json process_entries = json::array();
                for (int32_t i = 0; i < llama_process_functional_entry_count(ctx); ++i) {
                    llama_process_functional_entry_info info = {};
                    if (llama_process_functional_entry_get(ctx, i, &info) != 0 || !info.valid) {
                        continue;
                    }
                    const size_t blob_size = llama_process_functional_entry_blob_size(ctx, i);
                    if (blob_size == 0) {
                        continue;
                    }
                    std::vector<uint8_t> blob(blob_size);
                    if (llama_process_functional_entry_blob_export(ctx, i, blob.data(), blob.size()) != 0) {
                        throw std::runtime_error("failed to export process functional blob");
                    }
                    json entry = json::object();
                    entry["valid"] = info.valid;
                    entry["slot"] = info.slot;
                    entry["created_at_us"] = info.created_at_us;
                    entry["last_used_us"] = info.last_used_us;
                    entry["activation_count"] = info.activation_count;
                    entry["update_count"] = info.update_count;
                    entry["utility_score"] = info.utility_score;
                    entry["current_gain"] = info.current_gain;
                    entry["last_signed_outcome"] = info.last_signed_outcome;
                    entry["current_bootstrap_std"] = info.current_bootstrap_std;
                    entry["last_bootstrap_perturbation"] = info.last_bootstrap_perturbation;
                    entry["signature"] = {
                            {"valid", info.signature.valid},
                            {"signature_hash", info.signature.signature_hash},
                            {"scope_kind", info.signature.scope_kind},
                            {"family", info.signature.family},
                            {"loop_origin", info.signature.loop_origin},
                            {"microphase", info.signature.microphase},
                            {"plan_mode", info.signature.plan_mode},
                            {"plan_step_kind", info.signature.plan_step_kind},
                            {"tool_kind", info.signature.tool_kind},
                            {"source_family", info.signature.source_family},
                            {"requires_tool_result", info.signature.requires_tool_result},
                            {"transient_plan_id", info.signature.transient_plan_id},
                            {"transient_step_index", info.signature.transient_step_index},
                            {"transient_source_id", info.signature.transient_source_id},
                            {"tool_name", info.signature.tool_name},
                            {"capability_id", info.signature.capability_id},
                            {"provenance_namespace", info.signature.provenance_namespace},
                            {"semantic_key", info.signature.semantic_key},
                    };
                    entry["blob_b64"] = base64::encode((const char *) blob.data(), blob.size());
                    process_entries.push_back(std::move(entry));
                }
                snapshot["process_functional_entries"] = std::move(process_entries);

                json process_snapshots = json::array();
                for (int32_t entry_slot = 0; entry_slot < LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES; ++entry_slot) {
                    llama_process_functional_entry_info entry_info = {};
                    if (llama_process_functional_entry_get(ctx, entry_slot, &entry_info) != 0 || !entry_info.valid) {
                        continue;
                    }
                    llama_functional_lora_snapshot_archive archive = {};
                    if (llama_process_functional_snapshot_archive_get(ctx, entry_slot, &archive) != 0) {
                        throw std::runtime_error("failed to query process functional snapshot archive");
                    }

                    json entry_archive = json::object();
                    entry_archive["entry_slot"] = entry_slot;
                    entry_archive["family"] = archive.family;
                    entry_archive["count"] = archive.count;
                    entry_archive["last_capture_us"] = archive.last_capture_us;
                    entry_archive["next_capture_due_us"] = archive.next_capture_due_us;
                    json items = json::array();
                    for (int32_t snapshot_slot = 0; snapshot_slot < LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY; ++snapshot_slot) {
                        llama_functional_lora_snapshot_info info = {};
                        if (llama_process_functional_snapshot_info_get(ctx, entry_slot, snapshot_slot, &info) != 0) {
                            throw std::runtime_error("failed to query process functional snapshot info");
                        }
                        if (!info.valid) {
                            continue;
                        }
                        const size_t blob_size = llama_process_functional_snapshot_blob_size(ctx, entry_slot, snapshot_slot);
                        if (blob_size == 0) {
                            throw std::runtime_error("process functional snapshot archive reported empty blob");
                        }
                        std::vector<uint8_t> blob(blob_size);
                        if (llama_process_functional_snapshot_blob_export(ctx, entry_slot, snapshot_slot, blob.data(), blob.size()) != 0) {
                            throw std::runtime_error("failed to export process functional snapshot blob");
                        }

                        json item = json::object();
                        item["valid"] = info.valid;
                        item["family"] = info.family;
                        item["slot"] = info.slot;
                        item["source"] = info.source;
                        item["snapshot_id"] = info.snapshot_id;
                        item["captured_at_us"] = info.captured_at_us;
                        item["expires_at_us"] = info.expires_at_us;
                        item["source_update_count"] = info.source_update_count;
                        item["self_state_gradient_norm"] = info.self_state_gradient_norm;
                        item["robustness_score"] = info.robustness_score;
                        item["last_signed_outcome"] = info.last_signed_outcome;
                        item["dominant_direction_cosine"] = info.dominant_direction_cosine;
                        item["blob_b64"] = base64::encode((const char *) blob.data(), blob.size());
                        items.push_back(std::move(item));
                    }
                    entry_archive["items"] = std::move(items);
                    process_snapshots.push_back(std::move(entry_archive));
                }
                snapshot["process_functional_snapshots"] = std::move(process_snapshots);
            }

            const std::filesystem::path target_path(snapshot_path);
            if (!target_path.parent_path().empty()) {
                std::filesystem::create_directories(target_path.parent_path());
            }
            const std::filesystem::path tmp_path = target_path.string() + ".tmp";
            {
                std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
                if (!out) {
                    throw std::runtime_error("failed to open temporary runtime snapshot");
                }
                out << snapshot.dump(2);
                out.flush();
                if (!out) {
                    throw std::runtime_error("failed to flush runtime snapshot");
                }
            }
            if (std::filesystem::exists(target_path)) {
                std::filesystem::remove(target_path);
            }
            std::filesystem::rename(tmp_path, target_path);

            {
                std::lock_guard<std::mutex> lock(runtime_state_mutex);
                runtime_state_dirty = false;
                runtime_persistence.healthy = true;
                runtime_persistence.last_error.clear();
                runtime_persistence.last_persist_ms = ggml_time_ms();
                runtime_persistence.persist_success_total++;
            }
            SRV_INF("runtime snapshot persisted: path=%s reason=%s\n", snapshot_path.c_str(), reason ? reason : "unspecified");
            return true;
        } catch (const std::exception & err) {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            runtime_persistence.healthy = false;
            runtime_persistence.last_error = err.what();
            runtime_persistence.last_persist_ms = ggml_time_ms();
            runtime_persistence.persist_fail_total++;
            SRV_ERR("runtime snapshot persist failed: %s\n", err.what());
            return false;
        }
    }

    bool restore_runtime_state() {
        std::string snapshot_path;
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            if (!runtime_persistence.enabled || runtime_persistence.snapshot_path.empty()) {
                return true;
            }
            snapshot_path = runtime_persistence.snapshot_path;
            runtime_persistence.restore_attempted = true;
        }

        if (!std::filesystem::exists(snapshot_path)) {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            runtime_persistence.restore_success = false;
            runtime_persistence.healthy = true;
            runtime_persistence.last_error.clear();
            runtime_persistence.last_restore_ms = ggml_time_ms();
            return true;
        }

        try {
            std::ifstream in(snapshot_path, std::ios::binary);
            if (!in) {
                throw std::runtime_error("failed to open runtime snapshot");
            }
            std::stringstream buffer;
            buffer << in.rdbuf();
            const json snapshot = json::parse(buffer.str());
            const int snapshot_version = json_value(snapshot, "version", 0);
            if (snapshot_version != 1 &&
                    snapshot_version != 2 &&
                    snapshot_version != 3 &&
                    snapshot_version != 4 &&
                    snapshot_version != 5 &&
                    snapshot_version != 6 &&
                    snapshot_version != 7 &&
                    snapshot_version != 8) {
                throw std::runtime_error("unsupported runtime snapshot version");
            }

            llama_bash_tool_config bash_config = llama_bash_tool_default_config();
            llama_hard_memory_config hard_memory_config = llama_hard_memory_default_config();
            if (!bash_tool_config_from_json(snapshot.at("bash_tool_config"), &bash_config) ||
                !hard_memory_config_from_json(snapshot.at("hard_memory_config"), &hard_memory_config) ||
                llama_bash_tool_configure(ctx, &bash_config) != 0 ||
                llama_hard_memory_configure(ctx, hard_memory_config) != 0) {
                throw std::runtime_error("failed to restore runtime tool configuration");
            }

            const std::string updater_blob = base64::decode(snapshot.at("self_state_updater_program_b64").get<std::string>());
            if (updater_blob.size() != sizeof(llama_self_updater_program)) {
                throw std::runtime_error("invalid updater program snapshot payload");
            }
            llama_self_updater_program updater = {};
            std::memcpy(&updater, updater_blob.data(), sizeof(updater));
            if (llama_self_state_set_updater_program(ctx, updater) != 0) {
                throw std::runtime_error("failed to restore self-state updater program");
            }

            const std::string trace_blob = base64::decode(snapshot.at("self_state_trace_b64").get<std::string>());
            if (!trace_blob.empty()) {
                if (llama_self_state_trace_import(ctx, trace_blob.data(), trace_blob.size(), true) != 0) {
                    throw std::runtime_error("failed to import self-state trace");
                }
                const int32_t trace_count = llama_self_state_trace_count(ctx);
                if (trace_count > 0 && llama_self_state_replay_trace(ctx, trace_count) != 0) {
                    throw std::runtime_error("failed to replay restored self-state trace");
                }
            }

            if (snapshot.contains("model_extensions") && snapshot.at("model_extensions").is_array()) {
                for (const auto & entry : snapshot.at("model_extensions")) {
                    llama_self_model_extension_update update = llama_self_model_extension_default_update();
                    if (!model_extension_update_from_json(entry, &update) ||
                        llama_self_state_upsert_model_extension(ctx, update) != 0) {
                        throw std::runtime_error("failed to restore model extension");
                    }
                }
            }

            if (snapshot_version >= 2 && snapshot.contains("proactive_mailbox")) {
                if (!proactive_mailbox_from_json(snapshot.at("proactive_mailbox"), &proactive_mailbox)) {
                    throw std::runtime_error("failed to restore proactive mailbox");
                }
            }

            if (snapshot_version >= 7 && snapshot.contains("telegram_outbox")) {
                if (!telegram_outbox_from_json(snapshot.at("telegram_outbox"), &telegram_outbox)) {
                    throw std::runtime_error("failed to restore telegram outbox");
                }
            }

            if (snapshot_version >= 6 && snapshot.contains("telegram_dialogue")) {
                if (!telegram_dialogue_history_from_json(snapshot.at("telegram_dialogue"), &telegram_dialogue_history)) {
                    throw std::runtime_error("failed to restore telegram dialogue history");
                }
            }

            if (snapshot_version >= 8 && snapshot.contains("pending_approvals")) {
                const auto & pending_approvals_json = snapshot.at("pending_approvals");
                std::unordered_map<std::string, vicuna_pending_approval> restored_by_id;
                std::unordered_map<int32_t, std::string> restored_id_by_command;
                uint64_t next_ordinal = std::max<uint64_t>(
                        1,
                        json_value(pending_approvals_json, "next_ordinal", uint64_t(1)));
                if (pending_approvals_json.contains("items") && pending_approvals_json.at("items").is_array()) {
                    for (const auto & entry : pending_approvals_json.at("items")) {
                        vicuna_pending_approval approval = {};
                        if (!pending_approval_from_json(entry, &approval)) {
                            continue;
                        }
                        restored_id_by_command[approval.command_id] = approval.approval_id;
                        restored_by_id.emplace(approval.approval_id, std::move(approval));
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    pending_approvals_by_id = std::move(restored_by_id);
                    pending_approval_id_by_command = std::move(restored_id_by_command);
                    next_pending_approval_ordinal = next_ordinal;
                }
            } else {
                std::lock_guard<std::mutex> lock(runtime_state_mutex);
                pending_approvals_by_id.clear();
                pending_approval_id_by_command.clear();
                next_pending_approval_ordinal = 1;
            }

            if (snapshot_version >= 3 && snapshot.contains("functional_snapshots")) {
                const auto & families = snapshot.at("functional_snapshots");
                if (!families.is_array()) {
                    throw std::runtime_error("invalid functional snapshot archive payload");
                }
                for (const auto & family_entry : families) {
                    if (!family_entry.is_object()) {
                        throw std::runtime_error("invalid functional snapshot archive entry");
                    }
                    const int32_t family = json_value(family_entry, "family", -1);
                    if (family < 0 || family >= LLAMA_FUNCTIONAL_LORA_COUNT) {
                        throw std::runtime_error("invalid functional snapshot family");
                    }
                    if (!family_entry.contains("items") || !family_entry.at("items").is_array()) {
                        throw std::runtime_error("invalid functional snapshot item list");
                    }
                    for (const auto & item : family_entry.at("items")) {
                        if (!item.is_object()) {
                            throw std::runtime_error("invalid functional snapshot item");
                        }
                        llama_functional_lora_snapshot_info info = {};
                        info.valid = json_value(item, "valid", false);
                        info.family = json_value(item, "family", family);
                        info.slot = json_value(item, "slot", -1);
                        info.source = json_value(item, "source", 0);
                        info.snapshot_id = json_value(item, "snapshot_id", (uint64_t) 0);
                        info.captured_at_us = json_value(item, "captured_at_us", (uint64_t) 0);
                        info.expires_at_us = json_value(item, "expires_at_us", (uint64_t) 0);
                        info.source_update_count = json_value(item, "source_update_count", (uint64_t) 0);
                        info.self_state_gradient_norm = json_value(item, "self_state_gradient_norm", 0.0f);
                        info.robustness_score = json_value(item, "robustness_score", 0.0f);
                        info.last_signed_outcome = json_value(item, "last_signed_outcome", 0.0f);
                        info.dominant_direction_cosine = json_value(item, "dominant_direction_cosine", 0.0f);
                        if (!info.valid) {
                            continue;
                        }
                        if (info.family != family ||
                                info.slot < 0 ||
                                info.slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY ||
                                !item.contains("blob_b64")) {
                            throw std::runtime_error("functional snapshot metadata mismatch");
                        }
                        const std::string blob = base64::decode(item.at("blob_b64").get<std::string>());
                        if (blob.empty()) {
                            throw std::runtime_error("functional snapshot blob missing");
                        }
                        if (llama_functional_lora_snapshot_blob_import(
                                    ctx,
                                    family,
                                    info.slot,
                                    info,
                                    blob.data(),
                                    blob.size()) != 0) {
                            throw std::runtime_error("failed to restore functional snapshot blob");
                        }
                    }
                }
            }

            if (snapshot_version >= 4 && snapshot.contains("process_functional_params")) {
                llama_process_functional_params process_params = llama_process_functional_default_params();
                const auto & process_params_json = snapshot.at("process_functional_params");
                process_params.enabled = json_value(process_params_json, "enabled", process_params.enabled);
                process_params.max_entries = json_value(process_params_json, "max_entries", process_params.max_entries);
                process_params.min_observations = json_value(process_params_json, "min_observations", process_params.min_observations);
                process_params.noop_abs_ceiling = json_value(process_params_json, "noop_abs_ceiling", process_params.noop_abs_ceiling);
                process_params.weak_positive_ceiling = json_value(process_params_json, "weak_positive_ceiling", process_params.weak_positive_ceiling);
                process_params.mean_outcome_ceiling = json_value(process_params_json, "mean_outcome_ceiling", process_params.mean_outcome_ceiling);
                process_params.weak_or_worse_ratio_threshold = json_value(process_params_json, "weak_or_worse_ratio_threshold", process_params.weak_or_worse_ratio_threshold);
                process_params.creation_cooldown_updates = json_value(process_params_json, "creation_cooldown_updates", process_params.creation_cooldown_updates);
                process_params.utility_decay = json_value(process_params_json, "utility_decay", process_params.utility_decay);
                if (llama_process_functional_set_params(ctx, process_params) != 0) {
                    throw std::runtime_error("failed to restore process functional params");
                }
            }

            if (snapshot_version >= 4 && snapshot.contains("process_functional_entries")) {
                const auto & entries = snapshot.at("process_functional_entries");
                if (!entries.is_array()) {
                    throw std::runtime_error("invalid process functional entry payload");
                }
                for (const auto & entry : entries) {
                    if (!entry.is_object() || !entry.contains("signature") || !entry.contains("blob_b64")) {
                        throw std::runtime_error("invalid process functional entry");
                    }
                    llama_process_functional_entry_info info = {};
                    info.valid = json_value(entry, "valid", false);
                    info.slot = json_value(entry, "slot", -1);
                    info.created_at_us = json_value(entry, "created_at_us", (uint64_t) 0);
                    info.last_used_us = json_value(entry, "last_used_us", (uint64_t) 0);
                    info.activation_count = json_value(entry, "activation_count", (uint64_t) 0);
                    info.update_count = json_value(entry, "update_count", (uint64_t) 0);
                    info.utility_score = json_value(entry, "utility_score", 0.0f);
                    info.current_gain = json_value(entry, "current_gain", 0.0f);
                    info.last_signed_outcome = json_value(entry, "last_signed_outcome", 0.0f);
                    info.current_bootstrap_std = json_value(entry, "current_bootstrap_std", 0.0f);
                    info.last_bootstrap_perturbation = json_value(entry, "last_bootstrap_perturbation", 0.0f);
                    const auto & signature = entry.at("signature");
                    info.signature.valid = json_value(signature, "valid", false);
                    info.signature.signature_hash = json_value(signature, "signature_hash", (uint64_t) 0);
                    info.signature.scope_kind = json_value(signature, "scope_kind", 0);
                    info.signature.family = json_value(signature, "family", -1);
                    info.signature.loop_origin = json_value(signature, "loop_origin", -1);
                    info.signature.microphase = json_value(signature, "microphase", 0);
                    info.signature.plan_mode = json_value(signature, "plan_mode", 0);
                    info.signature.plan_step_kind = json_value(signature, "plan_step_kind", 0);
                    info.signature.tool_kind = json_value(signature, "tool_kind", 0);
                    info.signature.source_family = json_value(signature, "source_family", -1);
                    info.signature.requires_tool_result = json_value(signature, "requires_tool_result", false);
                    info.signature.transient_plan_id = json_value(signature, "transient_plan_id", -1);
                    info.signature.transient_step_index = json_value(signature, "transient_step_index", -1);
                    info.signature.transient_source_id = json_value(signature, "transient_source_id", -1);
                    std::snprintf(info.signature.tool_name, sizeof(info.signature.tool_name), "%s", json_value(signature, "tool_name", std::string()).c_str());
                    std::snprintf(info.signature.capability_id, sizeof(info.signature.capability_id), "%s", json_value(signature, "capability_id", std::string()).c_str());
                    std::snprintf(info.signature.provenance_namespace, sizeof(info.signature.provenance_namespace), "%s", json_value(signature, "provenance_namespace", std::string()).c_str());
                    std::snprintf(info.signature.semantic_key, sizeof(info.signature.semantic_key), "%s", json_value(signature, "semantic_key", std::string()).c_str());
                    const std::string blob = base64::decode(entry.at("blob_b64").get<std::string>());
                    if (!info.valid ||
                            info.slot < 0 ||
                            info.slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES ||
                            blob.empty() ||
                            llama_process_functional_entry_blob_import(ctx, info.slot, &info, blob.data(), blob.size()) != 0) {
                        throw std::runtime_error("failed to restore process functional entry");
                    }
                }
            }

            if (snapshot_version >= 5 && snapshot.contains("process_functional_snapshots")) {
                const auto & entries = snapshot.at("process_functional_snapshots");
                if (!entries.is_array()) {
                    throw std::runtime_error("invalid process functional snapshot archive payload");
                }
                for (const auto & entry_archive : entries) {
                    if (!entry_archive.is_object()) {
                        throw std::runtime_error("invalid process functional snapshot archive entry");
                    }
                    const int32_t entry_slot = json_value(entry_archive, "entry_slot", -1);
                    if (entry_slot < 0 || entry_slot >= LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES) {
                        throw std::runtime_error("invalid process functional snapshot entry slot");
                    }
                    if (!entry_archive.contains("items") || !entry_archive.at("items").is_array()) {
                        throw std::runtime_error("invalid process functional snapshot item list");
                    }
                    for (const auto & item : entry_archive.at("items")) {
                        if (!item.is_object()) {
                            throw std::runtime_error("invalid process functional snapshot item");
                        }
                        llama_functional_lora_snapshot_info info = {};
                        info.valid = json_value(item, "valid", false);
                        info.family = json_value(item, "family", -1);
                        info.slot = json_value(item, "slot", -1);
                        info.source = json_value(item, "source", 0);
                        info.snapshot_id = json_value(item, "snapshot_id", (uint64_t) 0);
                        info.captured_at_us = json_value(item, "captured_at_us", (uint64_t) 0);
                        info.expires_at_us = json_value(item, "expires_at_us", (uint64_t) 0);
                        info.source_update_count = json_value(item, "source_update_count", (uint64_t) 0);
                        info.self_state_gradient_norm = json_value(item, "self_state_gradient_norm", 0.0f);
                        info.robustness_score = json_value(item, "robustness_score", 0.0f);
                        info.last_signed_outcome = json_value(item, "last_signed_outcome", 0.0f);
                        info.dominant_direction_cosine = json_value(item, "dominant_direction_cosine", 0.0f);
                        if (!info.valid) {
                            continue;
                        }
                        if (info.family < 0 ||
                                info.family >= LLAMA_FUNCTIONAL_LORA_COUNT ||
                                info.slot < 0 ||
                                info.slot >= LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY ||
                                !item.contains("blob_b64")) {
                            throw std::runtime_error("process functional snapshot metadata mismatch");
                        }
                        const std::string blob = base64::decode(item.at("blob_b64").get<std::string>());
                        if (blob.empty()) {
                            throw std::runtime_error("process functional snapshot blob missing");
                        }
                        if (llama_process_functional_snapshot_blob_import(
                                    ctx,
                                    entry_slot,
                                    info.slot,
                                    info,
                                    blob.data(),
                                    blob.size()) != 0) {
                            throw std::runtime_error("failed to restore process functional snapshot blob");
                        }
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(runtime_state_mutex);
                runtime_state_dirty = false;
                runtime_persistence.healthy = true;
                runtime_persistence.restore_success = true;
                runtime_persistence.last_error.clear();
                runtime_persistence.last_restore_ms = ggml_time_ms();
            }
            SRV_INF("runtime snapshot restored: path=%s\n", snapshot_path.c_str());
            return true;
        } catch (const std::exception & err) {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            runtime_persistence.healthy = false;
            runtime_persistence.restore_success = false;
            runtime_persistence.last_error = err.what();
            runtime_persistence.last_restore_ms = ggml_time_ms();
            SRV_ERR("runtime snapshot restore failed: %s\n", err.what());
            return false;
        }
    }

    bool enqueue_external_work(
            vicuna_external_work_queue & queue,
            vicuna_external_work_item && work) {
        {
            std::lock_guard<std::mutex> lock(queue.mutex);
            if (queue.stop) {
                return false;
            }
            if (queue.inflight_ids.find(work.command_id) != queue.inflight_ids.end()) {
                return true;
            }
            queue.inflight_ids.insert(work.command_id);
            queue.pending.push_back(std::move(work));
        }
        queue.cv.notify_one();
        return true;
    }

    bool active_runner_waiting_on_tool(int32_t * out_command_id, int32_t * out_tool_kind) const {
        if (!ctx) {
            return false;
        }

        llama_cognitive_active_runner_status runner = {};
        if (llama_cognitive_active_runner_get(ctx, &runner) != 0 ||
            !runner.active ||
            !runner.waiting_on_tool ||
            runner.pending_command_id <= 0) {
            return false;
        }

        for (int32_t i = 0, n = llama_cognitive_command_count(ctx); i < n; ++i) {
            llama_cognitive_command command = {};
            if (llama_cognitive_command_get(ctx, i, &command) != 0) {
                continue;
            }
            if (command.command_id == runner.pending_command_id &&
                (command.status == LLAMA_COG_COMMAND_STATUS_PENDING || command.status == LLAMA_COG_COMMAND_STATUS_ACKED)) {
                if (out_command_id) {
                    *out_command_id = command.command_id;
                }
                if (out_tool_kind) {
                    *out_tool_kind = command.tool_kind;
                }
                return true;
            }
        }

        return false;
    }

    bool dispatch_pending_tool_commands(int32_t origin_filter) {
        if (!ctx) {
            return false;
        }

        auto submit_bash_error = [&](const llama_cognitive_command & command, const char * error_text) {
            llama_bash_tool_result result = {};
            result.command_id = command.command_id;
            result.tool_job_id = command.tool_job_id;
            result.launch_failed = true;
            result.exit_code = 127;
            std::snprintf(result.error_text, sizeof(result.error_text), "%s", error_text);
            if (llama_cognitive_bash_tool_submit_result(ctx, &result, nullptr) == 0) {
                mark_runtime_state_dirty("bash-tool-immediate-result");
                capture_tool_result_provenance("bash_tool_immediate", bash_result_to_json(result), nullptr);
            }
        };

        auto submit_hard_memory_error = [&](const llama_cognitive_command & command, int32_t status_code, const char * error_text) {
            llama_cognitive_hard_memory_result result = {};
            result.command_id = command.command_id;
            result.tool_job_id = command.tool_job_id;
            result.operation =
                    command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE ?
                            LLAMA_COG_HARD_MEMORY_OPERATION_WRITE :
                            LLAMA_COG_HARD_MEMORY_OPERATION_QUERY;
            if (result.operation == LLAMA_COG_HARD_MEMORY_OPERATION_WRITE) {
                result.archive_trace.tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_WRITE;
                result.archive_trace.status_code = status_code;
                result.archive_trace.request_started_us = ggml_time_us();
                result.archive_trace.request_completed_us = result.archive_trace.request_started_us;
                std::snprintf(result.archive_trace.error, sizeof(result.archive_trace.error), "%s", error_text);
            } else {
                result.result.ok = false;
                result.result.tool_kind = LLAMA_TOOL_KIND_HARD_MEMORY_QUERY;
                result.result.status_code = status_code;
                result.result.request_started_us = ggml_time_us();
                result.result.request_completed_us = result.result.request_started_us;
                std::snprintf(result.result.error, sizeof(result.result.error), "%s", error_text);
            }
            if (llama_cognitive_hard_memory_submit_result(ctx, &result, nullptr) == 0) {
                mark_runtime_state_dirty("hard-memory-immediate-result");
                capture_tool_result_provenance("hard_memory_immediate", hard_memory_result_to_json(result), nullptr);
            }
        };

        auto submit_codex_error = [&](const llama_cognitive_command & command, const char * error_text) {
            llama_codex_tool_result result = {};
            result.command_id = command.command_id;
            result.tool_job_id = command.tool_job_id;
            result.launch_failed = true;
            result.exit_code = 127;
            std::snprintf(result.error_text, sizeof(result.error_text), "%s", error_text);
            std::snprintf(result.summary_text, sizeof(result.summary_text), "%s", "Codex tool launch failed.");
            if (llama_cognitive_codex_tool_submit_result(ctx, &result, nullptr) == 0) {
                mark_runtime_state_dirty("codex-tool-immediate-result");
                capture_tool_result_provenance("codex_tool_immediate", codex_result_to_json(result), nullptr);
            }
        };

        auto submit_telegram_relay_result = [&](const llama_cognitive_command & command,
                                                const llama_telegram_relay_request * request,
                                                bool delivered,
                                                const char * error_text) {
            llama_telegram_relay_result result = {};
            result.command_id = command.command_id;
            result.tool_job_id = command.tool_job_id;
            result.intent_kind = request ? request->intent_kind : LLAMA_TELEGRAM_RELAY_COMMENT;
            result.delivered = delivered;
            result.delivered_at_ms = delivered ? ggml_time_ms() : 0;
            if (request) {
                std::snprintf(result.dedupe_key, sizeof(result.dedupe_key), "%s", request->dedupe_key);
            }
            if (error_text && error_text[0] != '\0') {
                std::snprintf(result.error_text, sizeof(result.error_text), "%s", error_text);
            }
            if (llama_cognitive_telegram_relay_submit_result(ctx, &result, nullptr) == 0) {
                mark_runtime_state_dirty("telegram-relay-immediate-result");
                capture_tool_result_provenance("telegram_relay_immediate", telegram_relay_result_to_json(result), nullptr);
            }
        };

        auto submit_telegram_ask_result = [&](const llama_cognitive_command & command,
                                              const llama_telegram_ask_options_request * request,
                                              bool delivered,
                                              const char * error_text) {
            llama_telegram_ask_options_result result = {};
            result.command_id = command.command_id;
            result.tool_job_id = command.tool_job_id;
            result.delivered = delivered;
            result.delivered_at_ms = delivered ? ggml_time_ms() : 0;
            if (request) {
                std::snprintf(result.dedupe_key, sizeof(result.dedupe_key), "%s", request->dedupe_key);
                std::snprintf(result.chat_scope, sizeof(result.chat_scope), "%s", request->chat_scope);
            }
            if (error_text && error_text[0] != '\0') {
                std::snprintf(result.error_text, sizeof(result.error_text), "%s", error_text);
            }
            if (llama_cognitive_telegram_ask_options_submit_result(ctx, &result, nullptr) == 0) {
                mark_runtime_state_dirty("telegram-ask-options-immediate-result");
                capture_tool_result_provenance("telegram_ask_options_immediate", telegram_ask_options_result_to_json(result), nullptr);
            }
        };

        bool dispatched = false;
        const int32_t command_count = llama_cognitive_command_count(ctx);
        for (int32_t i = 0; i < command_count; ++i) {
            llama_cognitive_command command = {};
            if (llama_cognitive_command_get(ctx, i, &command) != 0) {
                continue;
            }
            if (command.kind != LLAMA_COG_COMMAND_INVOKE_TOOL ||
                command.status != LLAMA_COG_COMMAND_STATUS_PENDING) {
                continue;
            }
            if (origin_filter >= 0 && command.origin != origin_filter) {
                continue;
            }

            const server_openclaw_capability_runtime * capability = nullptr;
            server_openclaw_dispatch_backend backend = SERVER_OPENCLAW_DISPATCH_NONE;
            if (openclaw_fabric.enabled()) {
                std::string resolve_error;
                capability = openclaw_fabric.resolve_command(command, &resolve_error);
                if (!capability) {
                    SRV_WRN("rejecting unresolved tool command %d: %s\n",
                            command.command_id,
                            resolve_error.c_str());
                    if (command.tool_kind == LLAMA_TOOL_KIND_BASH_CLI) {
                        submit_bash_error(command, resolve_error.c_str());
                    } else if (command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_QUERY ||
                               command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                        submit_hard_memory_error(command, 400, resolve_error.c_str());
                    } else if (command.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI) {
                        submit_codex_error(command, resolve_error.c_str());
                    } else if (command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY) {
                        submit_telegram_relay_result(command, nullptr, false, resolve_error.c_str());
                    } else {
                        (void) llama_cognitive_command_complete(ctx, command.command_id, true);
                    }
                    dispatched = true;
                    continue;
                }
                backend = capability->backend;
            } else if (command.tool_kind == LLAMA_TOOL_KIND_BASH_CLI) {
                backend = SERVER_OPENCLAW_DISPATCH_LEGACY_BASH;
            } else if (command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_QUERY ||
                       command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                backend = SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY;
            } else if (command.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI) {
                backend = SERVER_OPENCLAW_DISPATCH_LEGACY_CODEX;
            } else if (command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY ||
                       command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_ASK_OPTIONS) {
                backend = SERVER_OPENCLAW_DISPATCH_LEGACY_TELEGRAM;
            } else {
                continue;
            }

            bool approved_via_prompt = false;
            if (command_requires_user_approval(command, capability)) {
                const vicuna_pending_approval * approval = nullptr;
                std::string approval_id;
                {
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    approval = pending_approval_for_command_locked(command.command_id);
                    if (approval) {
                        approval_id = approval->approval_id;
                    }
                }

                if (!approval) {
                    std::string approval_error;
                    if (!ensure_command_approval_prompt(command, capability, &approval_error)) {
                        if (command.tool_kind == LLAMA_TOOL_KIND_BASH_CLI) {
                            submit_bash_error(command, approval_error.c_str());
                        } else if (command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_QUERY ||
                                   command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                            submit_hard_memory_error(command, 403, approval_error.c_str());
                        } else if (command.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI) {
                            submit_codex_error(command, approval_error.c_str());
                        } else if (command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY) {
                            submit_telegram_relay_result(command, nullptr, false, approval_error.c_str());
                        } else {
                            (void) llama_cognitive_command_complete(ctx, command.command_id, true);
                        }
                    } else {
                        dispatched = true;
                    }
                    continue;
                }

                if (approval->status == VICUNA_PENDING_APPROVAL_PENDING_PUBLISH ||
                        approval->status == VICUNA_PENDING_APPROVAL_AWAITING_USER) {
                    dispatched = true;
                    continue;
                }

                if (approval->status != VICUNA_PENDING_APPROVAL_APPROVED) {
                    const std::string detail =
                            approval->resolution_detail.empty() ?
                                    "operator disallowed command before execution" :
                                    approval->resolution_detail;
                    if (command.tool_kind == LLAMA_TOOL_KIND_BASH_CLI) {
                        submit_bash_error(command, detail.c_str());
                    } else if (command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_QUERY ||
                               command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                        submit_hard_memory_error(command, 403, detail.c_str());
                    } else if (command.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI) {
                        submit_codex_error(command, detail.c_str());
                    } else if (command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY) {
                        submit_telegram_relay_result(command, nullptr, false, detail.c_str());
                    } else {
                        (void) llama_cognitive_command_complete(ctx, command.command_id, true);
                    }
                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        pending_approval_id_by_command.erase(command.command_id);
                        if (!approval_id.empty()) {
                            pending_approvals_by_id.erase(approval_id);
                        }
                    }
                    dispatched = true;
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    if (!approval_id.empty()) {
                        pending_approvals_by_id.erase(approval_id);
                        pending_approval_id_by_command.erase(command.command_id);
                    }
                }
                approved_via_prompt = true;
            }

            json authorization = command_authorization_trace_json(command, capability);
            if (approved_via_prompt) {
                authorization["authorization_source"] = "explicit_approval";
            }

            if (llama_cognitive_command_ack(ctx, command.command_id) != 0) {
                SRV_WRN("failed to ack tool command %d\n", command.command_id);
                continue;
            }

            if (command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_RELAY) {
                llama_telegram_relay_request request = {};
                if (llama_cognitive_telegram_relay_get_request(ctx, command.command_id, &request) != 0) {
                    SRV_WRN("telegram relay command %d did not have a pending request\n", command.command_id);
                    submit_telegram_relay_result(command, nullptr, false, "telegram relay request was missing");
                    continue;
                }

                const std::string relay_text = trim_ascii_copy(request.text);
                if (!request.command_ready || relay_text.empty()) {
                    submit_telegram_relay_result(command, &request, false, "telegram relay request did not include any text");
                    continue;
                }

                capture_tool_call_provenance(
                        "telegram_relay",
                        command,
                        telegram_relay_request_to_json(request),
                        &authorization);

                std::string response_id;
                const char * source_tag = request.dedupe_key[0] != '\0' ? request.dedupe_key : "telegram-relay";
                const bool published = proactive_mailbox_publish(relay_text, source_tag, &response_id);
                if (!published) {
                    {
                        std::lock_guard<std::mutex> lock(proactive_mailbox.mutex);
                        proactive_mailbox.fail_total++;
                    }
                    submit_telegram_relay_result(command, &request, false, "telegram relay publish failed");
                    continue;
                }

                append_telegram_dialogue_assistant_turn(
                        VICUNA_TELEGRAM_DIALOGUE_SCOPE_BROADCAST,
                        relay_text,
                        command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN ? "telegram_relay_dmn" : "telegram_relay_active",
                        response_id,
                        request.dedupe_key);

                if (!admit_runtime_emit_text(
                            ctx,
                            relay_text,
                            command.origin,
                            LLAMA_FUNCTIONAL_MICROPHASE_NONE,
                            command.command_id,
                            -1,
                            LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP)) {
                    SRV_WRN("failed to admit telegram relay command %d into self-state\n", command.command_id);
                }

                submit_telegram_relay_result(command, &request, true, "");
                mark_runtime_state_dirty("telegram-relay-publish");
                dispatched = true;
                SRV_INF("delivered telegram relay command %d origin=%d job=%d response=%s dedupe=\"%s\"\n",
                        command.command_id,
                        command.origin,
                        request.tool_job_id,
                        response_id.c_str(),
                        request.dedupe_key);
                continue;
            }

            if (command.tool_kind == LLAMA_TOOL_KIND_TELEGRAM_ASK_OPTIONS) {
                llama_telegram_ask_options_request request = {};
                if (llama_cognitive_telegram_ask_options_get_request(ctx, command.command_id, &request) != 0) {
                    SRV_WRN("telegram ask command %d did not have a pending request\n", command.command_id);
                    submit_telegram_ask_result(command, nullptr, false, "telegram ask request was missing");
                    continue;
                }

                const std::string question = trim_ascii_copy(request.question);
                const std::string chat_scope = trim_ascii_copy(request.chat_scope);
                std::vector<std::string> options;
                const int32_t option_count = std::clamp(request.option_count, 0, (int32_t) LLAMA_TELEGRAM_ASK_MAX_OPTIONS);
                for (int32_t i = 0; i < option_count; ++i) {
                    const std::string label = trim_ascii_copy(request.options[i].label);
                    if (!label.empty()) {
                        options.push_back(label);
                    }
                }
                if (!request.command_ready || question.empty() || chat_scope.empty() || options.size() < 2) {
                    submit_telegram_ask_result(command, &request, false, "ask_with_options request was incomplete");
                    continue;
                }

                capture_tool_call_provenance(
                        "ask_with_options",
                        command,
                        telegram_ask_options_request_to_json(request),
                        &authorization);

                if (!telegram_outbox_publish_ask_options(chat_scope, request.dedupe_key, question, options)) {
                    submit_telegram_ask_result(command, &request, false, "telegram ask prompt enqueue failed");
                    continue;
                }

                append_telegram_dialogue_assistant_turn(
                        chat_scope,
                        question,
                        command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN ? "telegram_ask_options_dmn" : "telegram_ask_options_active",
                        std::string(),
                        request.dedupe_key);

                if (!admit_runtime_emit_text(
                            ctx,
                            question,
                            command.origin,
                            LLAMA_FUNCTIONAL_MICROPHASE_NONE,
                            command.command_id,
                            -1,
                            LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP)) {
                    SRV_WRN("failed to admit telegram ask command %d into self-state\n", command.command_id);
                }

                submit_telegram_ask_result(command, &request, true, "");
                mark_runtime_state_dirty("telegram-ask-options-publish");
                dispatched = true;
                SRV_INF("enqueued telegram ask-with-options command %d origin=%d job=%d scope=%s dedupe=\"%s\"\n",
                        command.command_id,
                        command.origin,
                        request.tool_job_id,
                        request.chat_scope,
                        request.dedupe_key);
                continue;
            }

            if (backend == SERVER_OPENCLAW_DISPATCH_LEGACY_BASH) {
                llama_bash_tool_request request = {};
                if (llama_cognitive_bash_tool_get_request(ctx, command.command_id, &request) != 0) {
                    SRV_WRN("bash tool command %d did not have a pending request\n", command.command_id);
                    (void) llama_cognitive_command_complete(ctx, command.command_id, true);
                    continue;
                }

                llama_bash_tool_config config = llama_bash_tool_default_config();
                (void) llama_bash_tool_get_config(ctx, &config);
                if (!config.enabled || !request.command_ready || request.command_text[0] == '\0') {
                    submit_bash_error(command, !config.enabled ? "bash tool is disabled" : "bash tool request did not include a command");
                    continue;
                }

                capture_tool_call_provenance(
                        "bash_tool",
                        command,
                        bash_request_to_json(request),
                        &authorization);

                vicuna_external_work_item work = {};
                work.command_id = command.command_id;
                work.origin = command.origin;
                work.tool_kind = command.tool_kind;
                work.kind = VICUNA_EXTERNAL_WORK_BASH;
                work.bash_request = request;
                work.enqueued_ms = ggml_time_ms();
                if (enqueue_external_work(bash_work, std::move(work))) {
                    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ||
                            command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
                        (void) llama_cognitive_command_begin_external_wait(ctx, command.command_id);
                    }
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    external_observability.bash_dispatch_total++;
                    dispatched = true;
                    SRV_INF("queued bash tool command %d origin=%d job=%d cmd=\"%s\" intent=\"%s\"\n",
                            command.command_id,
                            command.origin,
                            request.tool_job_id,
                            request.command_text,
                            request.intent_text);
                }
            } else if (backend == SERVER_OPENCLAW_DISPATCH_LEGACY_HARD_MEMORY) {
                llama_cognitive_hard_memory_request request = {};
                if (llama_cognitive_hard_memory_get_request(ctx, command.command_id, &request) != 0) {
                    SRV_WRN("hard-memory command %d did not have a pending request\n", command.command_id);
                    (void) llama_cognitive_command_complete(ctx, command.command_id, true);
                    continue;
                }

                llama_hard_memory_config config = llama_hard_memory_default_config();
                (void) llama_hard_memory_get_config(ctx, &config);
                const bool is_write = request.operation == LLAMA_COG_HARD_MEMORY_OPERATION_WRITE;
                const bool valid_request =
                        is_write ?
                                (config.enabled && config.archive_enabled && request.write_count > 0) :
                                (config.enabled && request.query.query[0] != '\0');
                if (!valid_request) {
                    submit_hard_memory_error(
                            command,
                            !config.enabled ? 503 : 400,
                            !config.enabled ?
                                    "hard memory is disabled" :
                                    (is_write ?
                                            (!config.archive_enabled ?
                                                    "hard memory archival is disabled" :
                                                    "hard memory write request did not include any memories") :
                                            "hard memory request did not include a query"));
                    continue;
                }

                capture_tool_call_provenance(
                        "hard_memory",
                        command,
                        hard_memory_request_to_json(request),
                        &authorization);

                vicuna_external_work_item work = {};
                work.command_id = command.command_id;
                work.origin = command.origin;
                work.tool_kind = command.tool_kind;
                work.kind = VICUNA_EXTERNAL_WORK_HARD_MEMORY;
                work.hard_memory_request = request;
                work.hard_memory_config = config;
                work.enqueued_ms = ggml_time_ms();
                if (enqueue_external_work(hard_memory_work, std::move(work))) {
                    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ||
                            command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
                        (void) llama_cognitive_command_begin_external_wait(ctx, command.command_id);
                    }
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    external_observability.hard_memory_dispatch_total++;
                    dispatched = true;
                    SRV_INF("queued hard-memory command %d origin=%d job=%d mode=%d query=\"%s\" write_count=%d\n",
                            command.command_id,
                            command.origin,
                            request.tool_job_id,
                            request.operation,
                            request.query.query,
                            request.write_count);
                }
            } else if (backend == SERVER_OPENCLAW_DISPATCH_LEGACY_CODEX) {
                llama_codex_tool_request request = {};
                if (llama_cognitive_codex_tool_get_request(ctx, command.command_id, &request) != 0) {
                    SRV_WRN("codex command %d did not have a pending request\n", command.command_id);
                    (void) llama_cognitive_command_complete(ctx, command.command_id, true);
                    continue;
                }

                llama_codex_tool_config config = llama_codex_tool_default_config();
                (void) llama_codex_tool_get_config(ctx, &config);
                if (!config.enabled || !request.command_ready || request.task_prompt[0] == '\0') {
                    submit_codex_error(command, !config.enabled ? "codex tool is disabled" : "codex tool request did not include a task");
                    continue;
                }

                capture_tool_call_provenance(
                        "codex_tool",
                        command,
                        codex_request_to_json(request),
                        &authorization);

                vicuna_external_work_item work = {};
                work.command_id = command.command_id;
                work.origin = command.origin;
                work.tool_kind = command.tool_kind;
                work.kind = VICUNA_EXTERNAL_WORK_CODEX;
                work.codex_request = request;
                work.enqueued_ms = ggml_time_ms();
                if (enqueue_external_work(codex_work, std::move(work))) {
                    if (command.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ||
                            command.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
                        (void) llama_cognitive_command_begin_external_wait(ctx, command.command_id);
                    }
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    external_observability.codex_dispatch_total++;
                    dispatched = true;
                    SRV_INF("queued codex command %d origin=%d job=%d\n",
                            command.command_id,
                            command.origin,
                            request.tool_job_id);
                }
            } else {
                if (command.tool_kind == LLAMA_TOOL_KIND_BASH_CLI) {
                    submit_bash_error(command, "registered capability has no supported executor backend");
                } else if (command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_QUERY ||
                           command.tool_kind == LLAMA_TOOL_KIND_HARD_MEMORY_WRITE) {
                    submit_hard_memory_error(command, 501, "registered capability has no supported executor backend");
                } else if (command.tool_kind == LLAMA_TOOL_KIND_CODEX_CLI) {
                    submit_codex_error(command, "registered capability has no supported executor backend");
                } else {
                    (void) llama_cognitive_command_complete(ctx, command.command_id, true);
                }
            }
        }

        return dispatched;
    }

    bool drain_completed_external_work(int32_t origin_filter) {
        if (!ctx) {
            return false;
        }

        bool drained = false;
        auto drain_queue = [&](vicuna_external_work_queue & queue, vicuna_external_work_kind kind) {
            while (true) {
                vicuna_external_work_result result = {};
                {
                    std::lock_guard<std::mutex> lock(queue.mutex);
                    if (queue.completed.empty()) {
                        break;
                    }
                    result = queue.completed.front();
                    queue.completed.pop_front();
                }
                if (origin_filter >= 0 && result.origin != origin_filter) {
                    std::lock_guard<std::mutex> lock(queue.mutex);
                    queue.completed.push_back(std::move(result));
                    break;
                }

                drained = true;
                llama_active_loop_trace active_trace = {};
                std::string active_tool_xml;
                if (result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    auto it = waiting_active_tasks.find(result.command_id);
                    if (it != waiting_active_tasks.end()) {
                        active_tool_xml = trim_ascii_copy(it->second.active_trace.authoritative_turn.tool_call_payload);
                    }
                }
                if (!active_tool_xml.empty()) {
                    const std::vector<llama_token> payload_tokens = common_tokenize(vocab, active_tool_xml, true, true);
                    if (!payload_tokens.empty()) {
                        (void) llama_cognitive_active_tool_emission_note(
                                ctx,
                                result.command_id,
                                payload_tokens.data(),
                                payload_tokens.size());
                    }
                }
                bool ok = false;
                bool codex_post_completion_ok = true;
                if (kind == VICUNA_EXTERNAL_WORK_BASH) {
                    ok = llama_cognitive_bash_tool_submit_result(
                            ctx,
                            &result.bash_result,
                            result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? &active_trace : nullptr) == 0;
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    external_observability.bash_complete_total++;
                    if (!ok || result.bash_result.launch_failed || result.bash_result.exit_code != 0 || result.bash_result.timed_out) {
                        external_observability.bash_fail_total++;
                    }
                } else if (kind == VICUNA_EXTERNAL_WORK_HARD_MEMORY) {
                    ok = llama_cognitive_hard_memory_submit_result(
                            ctx,
                            &result.hard_memory_result,
                            result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? &active_trace : nullptr) == 0;
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    external_observability.hard_memory_complete_total++;
                    const bool hard_memory_failed =
                            result.hard_memory_result.operation == LLAMA_COG_HARD_MEMORY_OPERATION_WRITE ?
                                    !result.hard_memory_result.archive_trace.archived :
                                    !result.hard_memory_result.result.ok;
                    if (!ok || hard_memory_failed) {
                        external_observability.hard_memory_fail_total++;
                    }
                } else {
                    codex_post_completion_ok = schedule_codex_rebuild_after_completion(
                            result.codex_request,
                            &result.codex_result);
                    ok = llama_cognitive_codex_tool_submit_result(
                            ctx,
                            &result.codex_result,
                            result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? &active_trace : nullptr) == 0;
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    external_observability.codex_complete_total++;
                    if (!ok || !codex_post_completion_ok || result.codex_result.launch_failed || result.codex_result.exit_code != 0) {
                        external_observability.codex_fail_total++;
                    }
                }

	                if (!ok) {
	                    SRV_WRN("failed to submit external work result for command %d kind=%d\n",
	                            result.command_id,
	                            (int) kind);
	                    continue;
	                }
	
	                mark_runtime_state_dirty(
                            kind == VICUNA_EXTERNAL_WORK_BASH ? "bash-tool-result" :
                            kind == VICUNA_EXTERNAL_WORK_HARD_MEMORY ? "hard-memory-result" :
                            "codex-tool-result");
	                if (kind == VICUNA_EXTERNAL_WORK_BASH) {
	                    capture_tool_result_provenance(
	                            "bash_tool",
	                            bash_result_to_json(result.bash_result),
	                            result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? &active_trace : nullptr);
	                } else if (kind == VICUNA_EXTERNAL_WORK_HARD_MEMORY) {
	                    capture_tool_result_provenance(
	                            "hard_memory",
	                            hard_memory_result_to_json(result.hard_memory_result),
	                            result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? &active_trace : nullptr);
	                } else {
	                    capture_tool_result_provenance(
	                            "codex_tool",
	                            codex_result_to_json(result.codex_result),
	                            result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE ? &active_trace : nullptr);
	                }
	
	                if (result.origin == LLAMA_COG_COMMAND_ORIGIN_ACTIVE) {
	                    server_task resumed_task;
                    bool found_task = false;
                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        auto it = waiting_active_tasks.find(result.command_id);
                        if (it != waiting_active_tasks.end()) {
                            resumed_task = std::move(it->second);
                            waiting_active_tasks.erase(it);
                            found_task = true;
                        }
                    }
                    if (found_task) {
                        resumed_task.active_trace = active_trace;
                        resumed_task.has_active_trace = true;
                        resumed_task.react_resuming_from_tool_result = true;
                        if (server_task_should_prepare_authoritative_react(resumed_task)) {
                            append_react_tool_result(resumed_task, result);
                            if (!prepare_react_prompt(resumed_task)) {
                                SRV_WRN("failed to prepare resumed ReAct prompt for command %d\n", result.command_id);
                            }
                        }
                        queue_tasks.post(std::move(resumed_task), true);
                    } else {
                        SRV_WRN("missing waiting active task for command %d\n", result.command_id);
                    }
                } else if (result.origin == LLAMA_COG_COMMAND_ORIGIN_DMN) {
                    server_task resumed_task;
                    bool found_task = false;
                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        auto it = waiting_dmn_tasks.find(result.command_id);
                        if (it != waiting_dmn_tasks.end()) {
                            resumed_task = std::move(it->second);
                            waiting_dmn_tasks.erase(it);
                            found_task = true;
                        }
                    }
                    if (found_task) {
                        resumed_task.react_resuming_from_tool_result = true;
                        if (resumed_task.has_dmn_trace) {
                            resumed_task.dmn_trace.observation.valid = true;
                        }
                        if (server_task_should_prepare_authoritative_react(resumed_task)) {
                            append_react_tool_result(resumed_task, result);
                            if (!prepare_react_prompt(resumed_task)) {
                                SRV_WRN("failed to prepare resumed DMN ReAct prompt for command %d\n", result.command_id);
                            }
                        }
                        queue_tasks.post(std::move(resumed_task), true);
                    } else {
                        SRV_WRN("missing waiting DMN task for command %d\n", result.command_id);
                    }
                }

                if (kind == VICUNA_EXTERNAL_WORK_BASH) {
                    SRV_INF("bash tool command %d origin=%d exit=%d signal=%d timeout=%d launch_failed=%d error=\"%s\" stderr=\"%s\" stdout=\"%s\"\n",
                            result.command_id,
                            result.origin,
                            result.bash_result.exit_code,
                            result.bash_result.term_signal,
                            result.bash_result.timed_out ? 1 : 0,
                            result.bash_result.launch_failed ? 1 : 0,
                            result.bash_result.error_text,
                            result.bash_result.stderr_text,
                            result.bash_result.stdout_text);
                } else if (kind == VICUNA_EXTERNAL_WORK_HARD_MEMORY) {
                    SRV_INF("hard-memory command %d origin=%d ok=%d status=%d count=%d\n",
                            result.command_id,
                            result.origin,
                            result.hard_memory_result.result.ok ? 1 : 0,
                            result.hard_memory_result.result.status_code,
                            result.hard_memory_result.result.result_count);
                } else {
                    SRV_INF("codex command %d origin=%d exit=%d launch_failed=%d repo_changed=%d rebuild_attempted=%d rebuild_succeeded=%d\n",
                            result.command_id,
                            result.origin,
                            result.codex_result.exit_code,
                            result.codex_result.launch_failed ? 1 : 0,
                            result.codex_result.repo_changed ? 1 : 0,
                            result.codex_result.rebuild_attempted ? 1 : 0,
                            result.codex_result.rebuild_succeeded ? 1 : 0);
                }
            }
        };

        drain_queue(bash_work, VICUNA_EXTERNAL_WORK_BASH);
        drain_queue(hard_memory_work, VICUNA_EXTERNAL_WORK_HARD_MEMORY);
        drain_queue(codex_work, VICUNA_EXTERNAL_WORK_CODEX);
        return drained;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(const common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;
        state = SERVER_STATE_LOADING_MODEL;
        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            waiting_active_tasks.clear();
            runtime_state_dirty = false;
            runtime_persistence.enabled = false;
            runtime_persistence.healthy = true;
            runtime_persistence.restore_attempted = false;
            runtime_persistence.restore_success = false;
            runtime_persistence.snapshot_path.clear();
            runtime_persistence.last_error.clear();
            provenance_repository.enabled = false;
            provenance_repository.healthy = true;
            provenance_repository.next_sequence = 1;
            provenance_repository.append_total = 0;
            provenance_repository.append_fail_total = 0;
            provenance_repository.active_loop_total = 0;
            provenance_repository.tool_result_total = 0;
            provenance_repository.dmn_total = 0;
            provenance_repository.discovered_increase_total = 0;
            provenance_repository.permanent_increase_total = 0;
            provenance_repository.allostatic_increase_total = 0;
            provenance_repository.functional_update_observed_total = 0;
            provenance_repository.process_update_observed_total = 0;
            provenance_repository.prune_total = 0;
            provenance_repository.last_append_ms = 0;
            provenance_repository.path.clear();
            provenance_repository.active_path.clear();
            provenance_repository.session_id.clear();
            provenance_repository.last_error.clear();
            provenance_repository.retention_ms = 48LL * 60LL * 60LL * 1000LL;
            provenance_repository.active_bucket_start_ms = 0;
            provenance_repository.has_last_snapshot = false;
            provenance_repository.last_snapshot = {};
        }
        proactive_mailbox_reset_live_stream();

        llama_init = common_init_from_params(params_base);

        model = llama_init->model();
        ctx   = llama_init->context();

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model);

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            SRV_INF("loading draft model '%s'\n", params_base.speculative.mparams_dft.path.c_str());

            const auto & params_spec = params_base.speculative;

            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx) : params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx);
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams_dft;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
        }

        std::string & mmproj_path = params_base.mmproj.path;
        if (!mmproj_path.empty()) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mtmd_context_params mparams = mtmd_context_params_default();

            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }

            if (params_base.speculative.type != COMMON_SPECULATIVE_TYPE_NONE) {
                params_base.speculative.type =  COMMON_SPECULATIVE_TYPE_NONE;
                SRV_WRN("%s\n", "speculative decoding is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        {
            const char * vicuna_active_lora = getenv("VICUNA_ACTIVE_LORA");
            const char * vicuna_past_lora = getenv("VICUNA_PAST_LORA");
            const bool enable_active_lora = vicuna_active_lora && atoi(vicuna_active_lora) != 0;
            const bool enable_past_lora = vicuna_past_lora && atoi(vicuna_past_lora) != 0;
            if (enable_active_lora || enable_past_lora) {
                llama_active_lora_params active_lora = llama_active_lora_default_params();
                active_lora.enabled = true;
                if (llama_active_lora_init(ctx, active_lora) == 0) {
                    SRV_INF("%s\n", "Active LoRA memory enabled");
                    if (enable_past_lora) {
                        llama_past_lora_params past_lora = llama_past_lora_default_params();
                        past_lora.enabled = true;
                        if (llama_past_lora_init(ctx, past_lora) == 0) {
                            SRV_INF("%s\n", "Past LoRA condensation stack enabled");
                        } else {
                            SRV_WRN("%s\n", "failed to initialize Past LoRA condensation stack");
                        }
                    }
                } else {
                    SRV_WRN("%s\n", "failed to initialize Active LoRA memory");
                }
            }
        }

        {
            const char * runtime_state_path = getenv("VICUNA_RUNTIME_STATE_PATH");
            if (runtime_state_path && runtime_state_path[0] != '\0') {
                std::lock_guard<std::mutex> lock(runtime_state_mutex);
                runtime_persistence.enabled = true;
                runtime_persistence.snapshot_path = runtime_state_path;
                SRV_INF("runtime persistence enabled: path=%s\n", runtime_state_path);
            }
        }

        {
            const char * provenance_enabled_env = getenv("VICUNA_PROVENANCE_ENABLED");
            const char * provenance_path_env = getenv("VICUNA_PROVENANCE_LOG_PATH");
            const char * provenance_retention_hours_env = getenv("VICUNA_PROVENANCE_RETENTION_HOURS");
            const bool runtime_state_enabled = runtime_persistence.enabled && !runtime_persistence.snapshot_path.empty();
            const bool provenance_enabled = parse_env_flag(
                    provenance_enabled_env,
                    (provenance_path_env && provenance_path_env[0] != '\0') || runtime_state_enabled);
            std::string provenance_path;
            if (provenance_path_env && provenance_path_env[0] != '\0') {
                provenance_path = provenance_path_env;
            } else if (runtime_state_enabled) {
                provenance_path = runtime_persistence.snapshot_path + ".provenance.jsonl";
            }

            if (provenance_enabled) {
                if (provenance_path.empty()) {
                    SRV_WRN("%s\n", "provenance repository requested but no path could be derived");
                } else {
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    provenance_repository.enabled = true;
                    provenance_repository.path = provenance_path;
                    if (provenance_retention_hours_env && provenance_retention_hours_env[0] != '\0') {
                        provenance_repository.retention_ms =
                                std::max<int64_t>(1, std::atoll(provenance_retention_hours_env)) * 60LL * 60LL * 1000LL;
                    }
                    provenance_repository.session_id = "prov_" + random_string();
                    provenance_repository.active_path.clear();
                    provenance_repository.active_bucket_start_ms = 0;
                    SRV_INF("provenance repository enabled: path=%s retention_ms=%lld session=%s\n",
                            provenance_repository.path.c_str(),
                            (long long) provenance_repository.retention_ms,
                            provenance_repository.session_id.c_str());
                }
            }
        }

        hard_memory_enabled = false;
        bash_tool_enabled = false;

        {
            if (const char * active_loop_env = getenv("VICUNA_ACTIVE_LOOP_ENABLED")) {
                active_loop_enabled = parse_env_flag(active_loop_env, active_loop_enabled);
            }
        }
        (void) configure_hard_memory_from_env(true);
        (void) configure_bash_tool_from_env(true);

        {
            llama_codex_tool_config codex_tool = llama_codex_tool_default_config();
            const char * codex_enabled = getenv("VICUNA_CODEX_TOOL_ENABLED");
            const char * codex_path = getenv("VICUNA_CODEX_TOOL_PATH");
            const char * codex_workdir = getenv("VICUNA_CODEX_TOOL_WORKDIR");
            const char * codex_timeout = getenv("VICUNA_CODEX_TOOL_TIMEOUT_MS");
            const char * codex_stdout = getenv("VICUNA_CODEX_TOOL_MAX_STDOUT_BYTES");
            const char * codex_stderr = getenv("VICUNA_CODEX_TOOL_MAX_STDERR_BYTES");
            const char * codex_rebuild = getenv("VICUNA_CODEX_TOOL_REBUILD_AFTER_CHANGES");
            const char * codex_verify = getenv("VICUNA_CODEX_TOOL_VERIFY_ACCESS_AFTER_REBUILD");
            const char * codex_rebuild_script = getenv("VICUNA_CODEX_TOOL_REBUILD_SCRIPT");
            const char * codex_rebuild_helper = getenv("VICUNA_CODEX_TOOL_REBUILD_HELPER");
            const char * codex_completion_message = getenv("VICUNA_CODEX_TOOL_COMPLETION_MESSAGE_PATH");

            if (codex_enabled) {
                codex_tool.enabled = atoi(codex_enabled) != 0;
            }
            if (codex_path) {
                std::snprintf(codex_tool.codex_path, sizeof(codex_tool.codex_path), "%s", codex_path);
            }
            if (codex_workdir) {
                std::snprintf(codex_tool.working_directory, sizeof(codex_tool.working_directory), "%s", codex_workdir);
            }
            if (codex_timeout) {
                codex_tool.timeout_ms = std::max(1000, atoi(codex_timeout));
            }
            if (codex_stdout) {
                codex_tool.max_stdout_bytes = std::max(1, atoi(codex_stdout));
            }
            if (codex_stderr) {
                codex_tool.max_stderr_bytes = std::max(1, atoi(codex_stderr));
            }
            if (codex_rebuild) {
                codex_tool.rebuild_after_changes = parse_env_flag(codex_rebuild, codex_tool.rebuild_after_changes);
            }
            if (codex_verify) {
                codex_tool.verify_tool_access_after_rebuild = parse_env_flag(codex_verify, codex_tool.verify_tool_access_after_rebuild);
            }
            if (codex_rebuild_script) {
                std::snprintf(codex_tool.rebuild_script_path, sizeof(codex_tool.rebuild_script_path), "%s", codex_rebuild_script);
            }
            if (codex_rebuild_helper) {
                std::snprintf(codex_tool.rebuild_helper_path, sizeof(codex_tool.rebuild_helper_path), "%s", codex_rebuild_helper);
            }
            if (codex_completion_message) {
                std::snprintf(codex_tool.completion_message_path, sizeof(codex_tool.completion_message_path), "%s", codex_completion_message);
            }

            if (llama_codex_tool_configure(ctx, &codex_tool) == 0) {
                configured_codex_tool = codex_tool;
                codex_tool_enabled = codex_tool.enabled && codex_tool.codex_path[0] != '\0' && codex_tool.working_directory[0] != '\0';
                if (codex_tool_enabled) {
                    SRV_INF("codex tool enabled: codex=%s cwd=%s timeout_ms=%d stdout=%d stderr=%d rebuild=%d verify=%d rebuild_script=%s helper=%s completion=%s\n",
                            codex_tool.codex_path,
                            codex_tool.working_directory,
                            codex_tool.timeout_ms,
                            codex_tool.max_stdout_bytes,
                            codex_tool.max_stderr_bytes,
                            codex_tool.rebuild_after_changes ? 1 : 0,
                            codex_tool.verify_tool_access_after_rebuild ? 1 : 0,
                            codex_tool.rebuild_script_path,
                            codex_tool.rebuild_helper_path,
                            codex_tool.completion_message_path);
                }
            } else {
                SRV_WRN("%s\n", "failed to configure codex tool");
            }
        }

        {
            std::string fabric_error;
            if (!openclaw_fabric.configure(bash_tool_enabled, hard_memory_enabled, codex_tool_enabled, &fabric_error)) {
                SRV_WRN("failed to configure OpenClaw tool fabric: %s\n", fabric_error.c_str());
            } else if (openclaw_fabric.enabled()) {
                authoritative_react_control_enabled =
                        llama_cognitive_authoritative_react_set_enabled(ctx, true) == 0;
                const char * catalog_path = getenv("VICUNA_OPENCLAW_TOOL_FABRIC_CATALOG_PATH");
                const bool catalog_exists =
                        catalog_path && catalog_path[0] != '\0' &&
                        std::filesystem::exists(catalog_path);
                SRV_INF("OpenClaw prerequisites: bash=%d hard_memory=%d codex=%d catalog_path=%s catalog_exists=%d\n",
                        bash_tool_enabled ? 1 : 0,
                        hard_memory_enabled ? 1 : 0,
                        codex_tool_enabled ? 1 : 0,
                        catalog_path && catalog_path[0] != '\0' ? catalog_path : "<unset>",
                        catalog_exists ? 1 : 0);
                std::vector<llama_cognitive_tool_spec> specs;
                if (openclaw_fabric.build_cognitive_specs(&specs) &&
                        !specs.empty() &&
                        llama_cognitive_tool_spec_set(ctx, specs.data(), (int32_t) specs.size()) == 0) {
                    std::ostringstream capability_summary;
                    const auto & capabilities = openclaw_fabric.capabilities();
                    for (size_t i = 0; i < capabilities.size(); ++i) {
                        if (i > 0) {
                            capability_summary << ", ";
                        }
                        capability_summary << capabilities[i].descriptor.tool_name
                                           << "=" << capabilities[i].descriptor.capability_id;
                    }
                    SRV_INF("OpenClaw tool fabric enabled with %zu capabilities\n", specs.size());
                    SRV_INF("OpenClaw capability set: %s\n", capability_summary.str().c_str());
                    SRV_INF("authoritative ReAct control: %d\n", authoritative_react_control_enabled ? 1 : 0);
                } else {
                    if ((int32_t) specs.size() > LLAMA_COGNITIVE_MAX_TOOL_SPECS) {
                        SRV_WRN("OpenClaw tool count %d exceeds LLAMA_COGNITIVE_MAX_TOOL_SPECS=%d\n",
                                (int32_t) specs.size(),
                                (int32_t) LLAMA_COGNITIVE_MAX_TOOL_SPECS);
                    }
                    SRV_WRN("%s\n", "failed to install OpenClaw tool catalog into cognitive runtime");
                }
            }
        }

        if (llama_model_n_swa(model) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model);

        int n_ctx_slot = llama_n_ctx_seq(ctx);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        const bool can_spec = common_speculative_is_compat(ctx);
        if (!can_spec) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot slot;

            slot.id    = i;
            slot.ctx   = ctx;
            slot.n_ctx = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            // try speculative decoding
            if (can_spec) {
                slot.spec = common_speculative_init(params_base.speculative, slot.ctx);
                if (slot.spec) {
                    if (mctx) {
                        SRV_ERR("%s\n", "speculative decoding is not supported with multimodal");
                        return false;
                    }
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                } else {
                    SLT_INF(slot, "%s", "speculative decoding context not initialized\n");
                }
            }

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();

            slots.push_back(std::move(slot));
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("slots debug = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_WRN("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_WRN("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_WRN("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_WRN("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_WRN("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;
        core_system_prompt = resolve_vicuna_core_system_prompt(params_base);
        core_system_prompt_prefix = compose_vicuna_core_system_prefix(core_system_prompt);
        core_system_prompt_prefix_tokens = core_system_prompt_prefix.empty() ?
                llama_tokens() :
                common_tokenize(vocab, core_system_prompt_prefix, false, true);

        if (!restore_runtime_state()) {
            SRV_WRN("%s", "runtime restore failed; health will report degraded persistence\n");
        }
        reapply_env_external_tool_policy_after_restore();
        inject_startup_self_emit_from_env();
        state = SERVER_STATE_READY;

        if (!is_resume) {
            return init();
        }

        start_external_workers();
        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(model != nullptr);
        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();
        start_external_workers();

        if (!core_system_prompt.empty()) {
            SRV_INF("core system prompt enabled: chars=%zu keep_tokens=%zu\n",
                    core_system_prompt.size(),
                    core_system_prompt_prefix_tokens.size());
        }

        // populate webui settings
        {
            if (!params_base.webui_config_json.empty()) {
                try {
                    json_webui_settings = json::parse(params_base.webui_config_json);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled (reasoning_budget == 0)
            // 2. The chat template supports it
            const bool enable_thinking = params_base.use_jinja && params_base.reasoning_budget != 0 && common_chat_templates_support_enable_thinking(chat_templates.get());
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* media_path            */ params_base.media_path,
            };
        }

        return true;
    }

    json build_health_json() const {
        if (ctx) {
            (void) llama_self_state_refresh_time(ctx);
        }
        llama_self_model_revision self_model_revision = {};
        llama_emotive_moment_revision emotive_moment_revision = {};
        llama_self_social_state_info social_state = {};
        llama_self_disturbance_state_info disturbance_state = {};
        llama_self_tool_state_info tool_state = {};
        llama_shared_cognitive_context_window shared_context_window = {};
        llama_cognitive_host_state cognitive_host_state = {};
        llama_cognitive_dmn_runner_status dmn_runner_state = {};
        llama_dmn_tick_trace last_dmn_trace = {};
        const bool have_self_model_revision =
                ctx && llama_self_state_get_self_model_revision(ctx, &self_model_revision) == 0 && self_model_revision.valid;
        const bool have_emotive_moment_revision =
                ctx && llama_self_state_get_emotive_moment_revision(ctx, &emotive_moment_revision) == 0 && emotive_moment_revision.valid;
        const bool have_social_state =
                ctx && llama_self_state_get_social_state(ctx, &social_state) == 0;
        const bool have_disturbance_state =
                ctx && llama_self_state_get_last_disturbance(ctx, &disturbance_state) == 0 && disturbance_state.valid;
        const bool have_tool_state =
                ctx && llama_self_state_get_tool_state(ctx, &tool_state) == 0;
        const bool have_shared_context_window =
                ctx && llama_shared_cognitive_context_get_window(ctx, &shared_context_window) == 0;
        const bool have_cognitive_host_state =
                ctx && llama_cognitive_get_host_state(ctx, &cognitive_host_state) == 0;
        const bool have_dmn_runner_state =
                ctx && llama_cognitive_dmn_runner_get(ctx, &dmn_runner_state) == 0;
        const bool have_last_dmn_trace =
                ctx &&
                llama_dmn_get_last_trace(ctx, &last_dmn_trace) == 0 &&
                (last_dmn_trace.tick_id > 0 ||
                 last_dmn_trace.deferred_for_foreground ||
                 last_dmn_trace.admitted ||
                 last_dmn_trace.pressure.total > 0.0f ||
                 last_dmn_trace.loop_state.terminal_reason != LLAMA_COG_TERMINAL_NONE);
        json health = {
            {"status", "ok"},
            {"state", state == SERVER_STATE_READY ? "ready" : "loading"},
            {"sleeping", sleeping},
            {"waiting_active_tasks", 0},
            {"foreground_runtime", {
                {"waiting_on_tool_result", 0},
                {"waiting_on_deferred_delivery", 0},
                {"waiting_emit_response", 0},
                {"waiting_post_tool_decision", 0},
            }},
            {"external_bash_pending", count_pending_external_work(bash_work)},
            {"external_hard_memory_pending", count_pending_external_work(hard_memory_work)},
            {"cognitive_state", {
                {"self_model_revision", {
                    {"valid", have_self_model_revision},
                    {"revision_id", have_self_model_revision ? self_model_revision.revision_id : 0},
                    {"allostatic_distance", have_self_model_revision ? self_model_revision.allostatic_distance : 0.0},
                    {"materiality_score", have_self_model_revision ? self_model_revision.materiality_score : 0.0},
                }},
                {"emotive_moment", {
                    {"valid", have_emotive_moment_revision},
                    {"revision_id", have_emotive_moment_revision ? emotive_moment_revision.revision_id : 0},
                    {"source_self_model_revision_id", have_emotive_moment_revision ? emotive_moment_revision.source_self_model_revision_id : 0},
                    {"materiality_score", have_emotive_moment_revision ? emotive_moment_revision.materiality_score : 0.0},
                    {"text", have_emotive_moment_revision ? bounded_cstr_to_string(emotive_moment_revision.text) : std::string()},
                }},
                {"social_state", {
                    {"valid", have_social_state},
                    {"bond_strength", have_social_state ? social_state.bond_strength : 0.0},
                    {"dissatisfaction", have_social_state ? social_state.dissatisfaction : 0.0},
                    {"contact_set_point_hours", have_social_state ? social_state.contact_set_point_hours : 0.0},
                    {"silence_hours", have_social_state ? social_state.silence_hours : 0.0},
                    {"silence_deficit", have_social_state ? social_state.silence_deficit : 0.0},
                }},
                {"tool_state", {
                    {"valid", have_tool_state},
                    {"active_status", have_tool_state ? tool_state.active_status : LLAMA_SELF_TOOL_JOB_IDLE},
                    {"pending_jobs", have_tool_state ? tool_state.pending_jobs : 0},
                    {"running_jobs", have_tool_state ? tool_state.running_jobs : 0},
                    {"completed_jobs", have_tool_state ? tool_state.completed_jobs : 0},
                    {"failed_jobs", have_tool_state ? tool_state.failed_jobs : 0},
                    {"readiness", have_tool_state ? tool_state.readiness : 0.0},
                    {"last_update_monotonic_ms", have_tool_state ? tool_state.last_update_monotonic_ms : -1},
                }},
                {"last_disturbance", {
                    {"valid", have_disturbance_state},
                    {"source_kind", have_disturbance_state ? disturbance_state.source_kind : 0},
                    {"failure_class", have_disturbance_state ? disturbance_state.failure_class : 0},
                    {"source_reliability", have_disturbance_state ? disturbance_state.source_reliability : 0.0},
                    {"total_disturbance", have_disturbance_state ? disturbance_state.delta.total_disturbance : 0.0},
                    {"social_deficit", have_disturbance_state ? disturbance_state.appraisal.social_deficit : 0.0},
                    {"expectation_violation", have_disturbance_state ? disturbance_state.appraisal.expectation_violation : 0.0},
                    {"progress_error", have_disturbance_state ? disturbance_state.appraisal.progress_error : 0.0},
                }},
                {"shared_context_window", {
                    {"valid", have_shared_context_window},
                    {"head_revision", have_shared_context_window ? shared_context_window.head_revision : 0},
                    {"item_count", have_shared_context_window ? shared_context_window.item_count : 0},
                    {"token_count", have_shared_context_window ? shared_context_window.token_count : 0},
                }},
                {"dmn_state", {
                    {"host_state_valid", have_cognitive_host_state},
                    {"runner_state_valid", have_dmn_runner_state},
                    {"last_trace_valid", have_last_dmn_trace},
                    {"host_state", {
                        {"shared_state_version", have_cognitive_host_state ? cognitive_host_state.shared_state_version : 0},
                        {"active_episode_count", have_cognitive_host_state ? cognitive_host_state.active_episode_count : 0},
                        {"dmn_tick_count", have_cognitive_host_state ? cognitive_host_state.dmn_tick_count : 0},
                        {"background_deferred_count", have_cognitive_host_state ? cognitive_host_state.background_deferred_count : 0},
                        {"pending_tool_followup_count", have_cognitive_host_state ? cognitive_host_state.pending_tool_followup_count : 0},
                        {"pending_dmn_emits", have_cognitive_host_state ? cognitive_host_state.pending_dmn_emits : 0},
                        {"last_foreground_time_us", have_cognitive_host_state ? cognitive_host_state.last_foreground_time_us : 0},
                        {"last_dmn_time_us", have_cognitive_host_state ? cognitive_host_state.last_dmn_time_us : 0},
                    }},
                    {"runner_state", {
                        {"tick_id", have_dmn_runner_state ? dmn_runner_state.tick_id : 0},
                        {"active", have_dmn_runner_state ? dmn_runner_state.active : false},
                        {"waiting_on_tool", have_dmn_runner_state ? dmn_runner_state.waiting_on_tool : false},
                        {"completed", have_dmn_runner_state ? dmn_runner_state.completed : false},
                        {"planning_active", have_dmn_runner_state ? dmn_runner_state.planning_active : false},
                        {"steps_taken", have_dmn_runner_state ? dmn_runner_state.steps_taken : 0},
                        {"max_steps", have_dmn_runner_state ? dmn_runner_state.max_steps : 0},
                        {"pending_command_id", have_dmn_runner_state ? dmn_runner_state.pending_command_id : 0},
                        {"pending_tool_spec_index", have_dmn_runner_state ? dmn_runner_state.pending_tool_spec_index : 0},
                        {"last_command_id", have_dmn_runner_state ? dmn_runner_state.last_command_id : 0},
                        {"functional_microphase", have_dmn_runner_state ? dmn_runner_state.functional_microphase : 0},
                        {"plan_id", have_dmn_runner_state ? dmn_runner_state.plan_id : 0},
                        {"plan_mode", have_dmn_runner_state ? dmn_runner_state.plan_mode : 0},
                        {"plan_status", have_dmn_runner_state ? dmn_runner_state.plan_status : 0},
                        {"plan_revision_count", have_dmn_runner_state ? dmn_runner_state.plan_revision_count : 0},
                        {"current_plan_step", have_dmn_runner_state ? dmn_runner_state.current_plan_step : 0},
                        {"self_model_revision_id", have_dmn_runner_state ? dmn_runner_state.self_model_revision_id : 0},
                        {"emotive_revision_id", have_dmn_runner_state ? dmn_runner_state.emotive_revision_id : 0},
                        {"context_revision", have_dmn_runner_state ? dmn_runner_state.context_revision : 0},
                        {"turn_id", have_dmn_runner_state ? dmn_runner_state.turn_id : 0},
                    }},
                    {"last_trace", have_last_dmn_trace ? dmn_trace_summary_to_json(last_dmn_trace) : json::object()},
                }},
            }},
            {"proactive_mailbox", build_proactive_mailbox_health_json()},
            {"telegram_outbox", build_telegram_outbox_health_json()},
            {"telegram_dialogue", build_telegram_dialogue_health_json()},
            {"foreground_consent", json{
                {"count", 0},
                {"items", json::array()},
            }},
            {"pending_approvals", json{
                {"count", 0},
                {"pending_publish", 0},
                {"awaiting_user", 0},
                {"approved_waiting_dispatch", 0},
                {"active_origin", 0},
                {"dmn_origin", 0},
                {"items", json::array()},
            }},
            {"runtime_persistence", {
                {"enabled", false},
                {"healthy", true},
                {"restore_attempted", false},
                {"restore_success", false},
                {"last_persist_ms", 0},
                {"last_restore_ms", 0},
                {"path", ""},
                {"last_error", ""},
            }},
            {"provenance_repository", {
                {"enabled", false},
                {"healthy", true},
                {"session_id", ""},
                {"path", ""},
                {"last_append_ms", 0},
                {"append_total", 0},
                {"append_fail_total", 0},
                {"active_loop_total", 0},
                {"tool_result_total", 0},
                {"dmn_total", 0},
                {"discovered_increase_total", 0},
                {"permanent_increase_total", 0},
                {"allostatic_increase_total", 0},
                {"functional_update_total", 0},
                {"process_update_total", 0},
                {"self_state", {
                    {"active_count", 0},
                    {"discovered_count", 0},
                    {"permanent_count", 0},
                    {"allostatic_count", 0},
                    {"allostatic_divergence", 0.0},
                    {"promotion_readiness", 0.0},
                    {"belief_pressure", 0.0},
                }},
                {"last_error", ""},
            }},
        };

        health["foreground_consent"] = build_foreground_consent_health_json();
        health["pending_approvals"] = build_pending_approvals_health_json();

        {
            std::lock_guard<std::mutex> lock(runtime_state_mutex);
            size_t waiting_on_tool_result = 0;
            size_t waiting_on_deferred_delivery = 0;
            size_t waiting_emit_response = 0;
            size_t waiting_post_tool_decision = 0;
            for (const auto & entry : waiting_active_tasks) {
                const server_task & task = entry.second;
                waiting_on_tool_result += 1;
                if (task.telegram_deferred_delivery) {
                    waiting_on_deferred_delivery += 1;
                }
                if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE) {
                    waiting_emit_response += 1;
                }
                if (task.react_tool_stage == SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL) {
                    waiting_post_tool_decision += 1;
                }
            }
            health["waiting_active_tasks"] = waiting_active_tasks.size();
            health["foreground_runtime"] = {
                {"waiting_on_tool_result", waiting_on_tool_result},
                {"waiting_on_deferred_delivery", waiting_on_deferred_delivery},
                {"waiting_emit_response", waiting_emit_response},
                {"waiting_post_tool_decision", waiting_post_tool_decision},
            };
            health["proactive_mailbox"] = build_proactive_mailbox_health_json();
            health["telegram_dialogue"] = build_telegram_dialogue_health_json();
            health["runtime_persistence"] = {
                {"enabled", runtime_persistence.enabled},
                {"healthy", runtime_persistence.healthy},
                {"restore_attempted", runtime_persistence.restore_attempted},
                {"restore_success", runtime_persistence.restore_success},
                {"last_persist_ms", runtime_persistence.last_persist_ms},
                {"last_restore_ms", runtime_persistence.last_restore_ms},
                {"path", runtime_persistence.snapshot_path},
                {"last_error", runtime_persistence.last_error},
            };
            health["provenance_repository"] = {
                {"enabled", provenance_repository.enabled},
                {"healthy", provenance_repository.healthy},
                {"session_id", provenance_repository.session_id},
                {"path", provenance_repository.path},
                {"active_path", provenance_repository.active_path},
                {"retention_ms", provenance_repository.retention_ms},
                {"last_append_ms", provenance_repository.last_append_ms},
                {"append_total", provenance_repository.append_total},
                {"append_fail_total", provenance_repository.append_fail_total},
                {"prune_total", provenance_repository.prune_total},
                {"active_loop_total", provenance_repository.active_loop_total},
                {"tool_result_total", provenance_repository.tool_result_total},
                {"dmn_total", provenance_repository.dmn_total},
                {"discovered_increase_total", provenance_repository.discovered_increase_total},
                {"permanent_increase_total", provenance_repository.permanent_increase_total},
                {"allostatic_increase_total", provenance_repository.allostatic_increase_total},
                {"functional_update_total", provenance_repository.functional_update_observed_total},
                {"process_update_total", provenance_repository.process_update_observed_total},
                {"self_state", {
                    {"active_count", provenance_repository.last_snapshot.active_extensions},
                    {"discovered_count", provenance_repository.last_snapshot.discovered_extensions},
                    {"permanent_count", provenance_repository.last_snapshot.permanent_extensions},
                    {"allostatic_count", provenance_repository.last_snapshot.allostatic_extensions},
                    {"allostatic_divergence", provenance_repository.last_snapshot.allostatic_divergence},
                    {"promotion_readiness", provenance_repository.last_snapshot.promotion_readiness},
                    {"belief_pressure", provenance_repository.last_snapshot.belief_pressure},
                }},
                {"last_error", provenance_repository.last_error},
            };
            if (runtime_persistence.enabled && !runtime_persistence.healthy) {
                health["status"] = "degraded";
            }
            if (provenance_repository.enabled && !provenance_repository.healthy) {
                health["status"] = "degraded";
            }
        }

        return health;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            // don't update the cache if the slot's context is empty
            update_cache = update_cache && !tokens.empty();

            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_INF(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_INF(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        {
            if (task.has_active_trace) {
                SLT_INF(slot, "active authoritative episode %d deferred_background=%d\n",
                        task.active_trace.episode_id,
                        task.active_trace.deferred_background ? 1 : 0);
                llama_cognitive_active_runner_status runner = {};
                llama_cognitive_command command = {};
                if (llama_cognitive_active_runner_get(ctx, &runner) == 0 && runner.pending_command_id > 0) {
                    for (int32_t i = 0, n = llama_cognitive_command_count(ctx); i < n; ++i) {
                        if (llama_cognitive_command_get(ctx, i, &command) == 0 &&
                                command.command_id == runner.pending_command_id) {
                            SLT_INF(slot, "active runner episode %d command=%d kind=%d status=%d\n",
                                    runner.episode_id,
                                    command.command_id,
                                    command.kind,
                                    command.status);
                            break;
                        }
                    }
                }
            }
        }

        // initialize samplers
        if (task.need_sampling()) {
            slot.smpl.reset(common_sampler_init(model, task.params.sampling));

            if (slot.smpl == nullptr) {
                // for now, the only error that may happen here is invalid grammar
                send_error(task, "Failed to parse grammar", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_logits = task.params.sampling.n_probs > 0;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.spec && task.params.speculative.n_max > 0);

            // TODO: getting post/pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx, slot.id, nullptr);
            }

            SLT_INF(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    std::string deferred_active_error_followup_text(const std::string & error) const {
        const std::string trimmed = trim_ascii_copy(error);
        if (trimmed.empty()) {
            return "I ran into a problem while working on that request and could not complete it.";
        }
        return "I ran into a problem while working on that request and could not complete it: " + trimmed;
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        if (task_wants_deferred_telegram_delivery(task)) {
            (void) publish_deferred_telegram_delivery(
                    task,
                    deferred_active_error_followup_text(error),
                    "telegram_relay_active_error",
                    "error");
        }
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        if (slot.task && task_wants_deferred_telegram_delivery(*slot.task)) {
            (void) publish_deferred_telegram_delivery(
                    *slot.task,
                    deferred_active_error_followup_text(error),
                    "telegram_relay_active_error",
                    "error");
        }
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded           = slot.n_decoded;
        res->n_prompt_tokens     = slot.task->n_tokens();
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        if (slot.task) {
            SLT_INF(slot,
                    "authoritative react finalize: type=%d prepare=%d ready=%d origin=%d role=%d active=%d dmn=%d prefill=%zu tools=%zu retry=%d\n",
                    (int) slot.task->type,
                    server_task_should_prepare_authoritative_react(*slot.task) ? 1 : 0,
                    server_task_has_authoritative_react_surface(*slot.task) ? 1 : 0,
                    (int) slot.task->react_origin,
                    (int) slot.task->foreground_role,
                    slot.task->has_active_trace ? 1 : 0,
                    slot.task->has_dmn_trace ? 1 : 0,
                    slot.task->react_assistant_prefill.size(),
                    slot.task->react_tools.size(),
                    slot.task->react_retry_count);
        }
        parsed_react_step react_step = {};
        const bool parsed_react =
                slot.task &&
                server_task_has_authoritative_react_surface(*slot.task) &&
                parse_authoritative_react_step(*slot.task, slot.generated_text, &react_step);
        auto retry_staged_react = [&](server_task & retry_task, const std::string & failure_class, const std::string & failure_detail) -> bool {
            const std::string previous_failure_class = retry_task.react_last_failure_class;
            const std::string previous_failure_detail = retry_task.react_last_failure_detail;
            const bool repeated_same_failure =
                    !failure_class.empty() &&
                    failure_class == previous_failure_class &&
                    failure_detail == previous_failure_detail;
            retry_task.react_last_failure_class = failure_class;
            retry_task.react_last_failure_detail = failure_detail;
            retry_task.react_same_failure_count =
                    repeated_same_failure ?
                            retry_task.react_same_failure_count + 1 :
                            1;
            if (retry_task.react_retry_count >= retry_task.react_retry_limit) {
                return false;
            }

            retry_task.react_retry_count += 1;
            retry_task.react_stage_retry_count += 1;
            retry_task.react_retry_feedback = failure_detail;

            if (failure_class == "parse_failure" && slot.stop == STOP_TYPE_LIMIT) {
                const int32_t configured_predict =
                        retry_task.params.n_predict > 0 ?
                                retry_task.params.n_predict :
                                params_base.n_predict;
                const int32_t retry_predict =
                        configured_predict > 0 ?
                                std::max<int32_t>(192, std::min<int32_t>(512, configured_predict * 2)) :
                                192;
                retry_task.params.n_predict = retry_predict;
                retry_task.react_retry_feedback =
                        "The previous control step ran out of generation budget before completing the authoritative artifact. "
                        "Finish the same turn concisely with one Action line and the needed visible reply or tool payload.";
            }

            const bool read_only_external_action_conflict =
                    failure_class == "terminal_policy_reject" &&
                    failure_detail == "latest completed tool step was read-only, but the original request still requires an external action; continue the staged tool protocol";
            if (read_only_external_action_conflict) {
                retry_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_SELECT_TOOL;
                retry_task.react_stage_retry_count = 0;
                retry_task.react_selected_tool_family_id.clear();
                retry_task.react_selected_method_name.clear();
                retry_task.react_selected_capability_id.clear();
                retry_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                retry_task.react_retry_feedback =
                        "The previous completed tool step only verified state. "
                        "The original request still requires an external side effect. "
                        "Select the next tool family now instead of answering directly.";
            } else if (failure_class == "terminal_policy_reject") {
                retry_task.react_retry_feedback = failure_detail + ". Continue the same authoritative ReAct turn. " +
                        std::string(react_turn_requires_retry_tool_escalation(retry_task) ?
                                "This mutable turn has already failed without a tool. Continue the staged tool protocol immediately. "
                                "Do not answer directly or narrate intended work. " :
                                "If fresh external grounding is needed, continue the staged tool protocol. "
                                "Do not narrate intended work as the final answer.");
            }

            if (retry_task.react_same_failure_count >= 2 &&
                    failure_class == "terminal_policy_reject" &&
                    retry_task.react_resuming_from_tool_result) {
                retry_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_SELECT_TOOL;
                retry_task.react_stage_retry_count = 0;
                retry_task.react_selected_tool_family_id.clear();
                retry_task.react_selected_method_name.clear();
                retry_task.react_selected_capability_id.clear();
                retry_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                retry_task.react_retry_feedback =
                        "The previous post-tool continuation repeated the same rejected terminal choice. "
                        "Use the admitted tool observation and preserved staged history to select the next tool family now or produce a grounded final reply.";
            }

            if (retry_task.react_stage_retry_count > retry_task.react_stage_retry_limit) {
                retry_task.react_stage_retry_count = 0;
                if (retry_task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL) {
                    retry_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_SELECT_METHOD;
                    retry_task.react_selected_capability_id.clear();
                    retry_task.react_selected_method_name.clear();
                    retry_task.react_retry_feedback =
                            "The previous emit_arguments stage exceeded its retry budget. Select the appropriate method again based on the original request and current context.";
                } else if (retry_task.react_tool_stage == SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE) {
                    retry_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL;
                    retry_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                    retry_task.react_retry_feedback =
                            "The previous emit_response stage exceeded its retry budget. Select the appropriate next step again based on the original request, admitted tool observation, and current context.";
                } else if (retry_task.react_tool_stage == SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL) {
                    retry_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_SELECT_TOOL;
                    retry_task.react_selected_tool_family_id.clear();
                    retry_task.react_selected_method_name.clear();
                    retry_task.react_selected_capability_id.clear();
                    retry_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                    retry_task.react_retry_feedback =
                            "The previous decide_after_tool stage exceeded its retry budget. Select the appropriate tool again based on the original request, admitted tool observation, and current context.";
                } else if (retry_task.react_tool_stage == SERVER_REACT_TOOL_STAGE_SELECT_METHOD) {
                    retry_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_SELECT_TOOL;
                    retry_task.react_selected_tool_family_id.clear();
                    retry_task.react_selected_method_name.clear();
                    retry_task.react_selected_capability_id.clear();
                    retry_task.react_retry_feedback =
                            "The previous select_method stage exceeded its retry budget. Select the appropriate tool again based on the original request and current context.";
                }
            }

            if (retry_task.react_same_failure_count >= 3) {
                SRV_WRN("authoritative ReAct aborting repeated failure loop: task=%d stage=%d class=%s detail=%s count=%d\n",
                        retry_task.id,
                        retry_task.react_tool_stage,
                        failure_class.c_str(),
                        failure_detail.c_str(),
                        retry_task.react_same_failure_count);
                return false;
            }

            if (!prepare_react_prompt(retry_task)) {
                return false;
            }
            queue_tasks.post(std::move(retry_task), true);
            return true;
        };

        if (slot.task && server_task_has_authoritative_react_surface(*slot.task) && !parsed_react) {
            server_task retry_task = std::move(*slot.task);
            const std::string react_parse_input =
                    retry_task.react_assistant_prefill.empty() ?
                            slot.generated_text :
                            retry_task.react_assistant_prefill + slot.generated_text;
            SRV_WRN("authoritative ReAct parse failure: task=%d origin=%d retry=%d error=%s\nraw=%s\n",
                    retry_task.id,
                    (int) retry_task.react_origin,
                    retry_task.react_retry_count,
                    react_step.error.c_str(),
                    log_excerpt(react_parse_input).c_str());
            capture_react_stage_provenance(
                    retry_task,
                    "react_parse_failure",
                    "parse_failure",
                    nullptr,
                    react_parse_input,
                    react_step.error);
            if (retry_staged_react(retry_task, "parse_failure", react_step.error)) {
                return;
            }
            send_error(slot, "Invalid authoritative ReAct control: " + react_step.error, ERROR_TYPE_SERVER);
            return;
        }

        if (slot.task && server_task_has_authoritative_react_surface(*slot.task) && parsed_react) {
            SLT_INF(slot,
                    "authoritative react parsed: action=%d source=%s visible=%zu reasoning=%zu tool_calls=%zu xml=%zu\n",
                    react_step.action,
                    react_step.action_source.empty() ? "unspecified" : react_step.action_source.c_str(),
                    react_step.assistant_msg.content.size(),
                    react_step.assistant_msg.reasoning_content.size(),
                    react_step.assistant_msg.tool_calls.size(),
                    react_step.tool_xml.size());
            server_task resumed_task = std::move(*slot.task);
            resumed_task.react_retry_feedback.clear();
            resumed_task.react_last_failure_class.clear();
            resumed_task.react_last_failure_detail.clear();
            resumed_task.react_same_failure_count = 0;
            const std::string planner_reasoning = trim_ascii_copy(react_step.planner_reasoning);
            const std::string tool_xml_payload = trim_ascii_copy(react_step.tool_xml);
            const std::string react_parse_input =
                    resumed_task.react_assistant_prefill.empty() ?
                            slot.generated_text :
                            resumed_task.react_assistant_prefill + slot.generated_text;
            capture_react_stage_provenance(
                    resumed_task,
                    "react_stage_parsed",
                    "parsed",
                    &react_step,
                    react_parse_input);
            if (!planner_reasoning.empty()) {
                const char * reasoning_phase = react_tool_stage_name(resumed_task.react_tool_stage);
                if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_TOOL) {
                    reasoning_phase = "select_tool_family";
                } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_METHOD) {
                    reasoning_phase = "select_method";
                } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_EMIT_TOOL_CALL) {
                    reasoning_phase = "emit_arguments";
                } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_DECIDE_AFTER_TOOL) {
                    reasoning_phase = "decide_after_tool";
                } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_EMIT_RESPONSE) {
                    reasoning_phase = "emit_response";
                }
                SLT_INF(slot,
                        "authoritative react reasoning: phase=%s text=\n%s\n",
                        reasoning_phase,
                        bounded_runtime_log_excerpt(planner_reasoning, 4000).c_str());
            }
            const char * history_phase = react_tool_stage_name(resumed_task.react_tool_stage);
            if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_TOOL) {
                history_phase = "select_tool_family";
            } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_METHOD) {
                history_phase = "select_method";
            } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_EMIT_TOOL_CALL) {
                history_phase = "emit_arguments";
            } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_DECIDE_AFTER_TOOL) {
                history_phase = "decide_after_tool";
            } else if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_EMIT_RESPONSE) {
                history_phase = "emit_response";
            }
            if (!planner_reasoning.empty()) {
                append_react_history_entry(
                        resumed_task,
                        history_phase,
                        "reasoning",
                        std::string("<think>\n") + planner_reasoning + "\n</think>");
            }
            if (!tool_xml_payload.empty()) {
                append_react_history_entry(
                        resumed_task,
                        history_phase,
                        "stage_payload",
                        tool_xml_payload);
            }
            if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_TOOL) {
                record_react_last_step(
                        resumed_task,
                        "select_tool_family",
                        react_step.tool_xml,
                        react_json_text({
                                {"tool_family_id", react_step.selected_tool_family_id},
                        }));
                resumed_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_SELECT_METHOD;
                resumed_task.react_stage_retry_count = 0;
                resumed_task.react_selected_tool_family_id = react_step.selected_tool_family_id;
                resumed_task.react_selected_method_name.clear();
                resumed_task.react_selected_capability_id.clear();
                resumed_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                if (prepare_react_prompt(resumed_task)) {
                    capture_react_stage_provenance(
                            resumed_task,
                            "react_stage_transition",
                            "selected_tool",
                            &react_step,
                            react_parse_input);
                    queue_tasks.post(std::move(resumed_task), true);
                    return;
                }
                send_error(slot, "Failed to continue staged tool protocol after tool selection", ERROR_TYPE_SERVER);
                return;
            }
            if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_SELECT_METHOD) {
                record_react_last_step(
                        resumed_task,
                        "select_method",
                        react_step.tool_xml,
                        react_json_text({
                                {"tool_family_id", react_step.selected_tool_family_id},
                                {"method_name", react_step.selected_tool_method_name},
                                {"capability_id", react_step.selected_capability_id},
                        }));
                resumed_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_EMIT_TOOL_CALL;
                resumed_task.react_stage_retry_count = 0;
                resumed_task.react_selected_tool_family_id = react_step.selected_tool_family_id;
                resumed_task.react_selected_method_name = react_step.selected_tool_method_name;
                resumed_task.react_selected_capability_id = react_step.selected_capability_id;
                resumed_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                if (prepare_react_prompt(resumed_task)) {
                    capture_react_stage_provenance(
                            resumed_task,
                            "react_stage_transition",
                            "selected_method",
                            &react_step,
                            react_parse_input);
                    queue_tasks.post(std::move(resumed_task), true);
                    return;
                }
                send_error(slot, "Failed to continue staged tool protocol after method selection", ERROR_TYPE_SERVER);
                return;
            }
            if (react_step.tool_protocol_signal == REACT_TOOL_PROTOCOL_SIGNAL_DECIDE_AFTER_TOOL) {
                record_react_last_step(
                        resumed_task,
                        "decide_after_tool",
                        react_step.tool_xml,
                        react_json_text({
                                {"decision", react_terminal_action_label(react_step.action)},
                        }));
                resumed_task.react_stage_retry_count = 0;
                resumed_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                if (react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ACT) {
                    resumed_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_SELECT_TOOL;
                    resumed_task.react_selected_tool_family_id.clear();
                    resumed_task.react_selected_method_name.clear();
                    resumed_task.react_selected_capability_id.clear();
                    if (prepare_react_prompt(resumed_task)) {
                        capture_react_stage_provenance(
                                resumed_task,
                                "react_stage_transition",
                                "decided_select_tool",
                                &react_step,
                                react_parse_input);
                        queue_tasks.post(std::move(resumed_task), true);
                        return;
                    }
                    send_error(slot, "Failed to continue staged tool protocol after post-tool select_tool decision", ERROR_TYPE_SERVER);
                    return;
                }
                if (react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER ||
                        react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ASK) {
                    resumed_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_EMIT_RESPONSE;
                    resumed_task.react_pending_terminal_action = react_step.action;
                    if (prepare_react_prompt(resumed_task)) {
                        capture_react_stage_provenance(
                                resumed_task,
                                "react_stage_transition",
                                "decided_emit_response",
                                &react_step,
                                react_parse_input);
                        queue_tasks.post(std::move(resumed_task), true);
                        return;
                    }
                    send_error(slot, "Failed to continue staged response protocol after post-tool terminal decision", ERROR_TYPE_SERVER);
                    return;
                }
            }
            const std::string continuation_error =
                    validate_authoritative_react_terminal_policy(resumed_task, react_step);
            if (!continuation_error.empty()) {
                SRV_WRN("authoritative ReAct continuation reject: task=%d origin=%d retry=%d action=%d error=%s\n",
                        resumed_task.id,
                        (int) resumed_task.react_origin,
                        resumed_task.react_retry_count,
                        react_step.action,
                        continuation_error.c_str());
                capture_react_stage_provenance(
                        resumed_task,
                        "react_stage_reject",
                        "rejected",
                        &react_step,
                        react_parse_input,
                        continuation_error);
                const bool retried = retry_staged_react(
                        resumed_task,
                        "terminal_policy_reject",
                        continuation_error);
                if (retried) {
                    return;
                }
                send_error(slot, "Failed to continue authoritative ReAct turn after unsupported terminal step", ERROR_TYPE_SERVER);
                return;
            }
            llama_shared_cognitive_context_window context_window = {};
            if (!planner_reasoning.empty()) {
                const int32_t source_id =
                        resumed_task.react_origin == SERVER_REACT_ORIGIN_DMN ?
                                resumed_task.dmn_trace.tick_id :
                                resumed_task.active_trace.episode_id;
                const int32_t plan_id =
                        resumed_task.react_origin == SERVER_REACT_ORIGIN_DMN ?
                                resumed_task.dmn_trace.plan.plan_id :
                                resumed_task.active_trace.plan.plan_id;
                const int32_t loop_origin =
                        resumed_task.react_origin == SERVER_REACT_ORIGIN_DMN ?
                                LLAMA_COG_COMMAND_ORIGIN_DMN :
                                LLAMA_COG_COMMAND_ORIGIN_ACTIVE;
                if (!react_step.planner_reasoning_blocks.empty()) {
                    for (const std::string & block : react_step.planner_reasoning_blocks) {
                        (void) admit_runtime_emit_text(
                                ctx,
                                block,
                                loop_origin,
                                LLAMA_COG_LOOP_PHASE_PROPOSE,
                                source_id,
                                plan_id,
                                0,
                                LLAMA_SELF_COG_ARTIFACT_HIDDEN_THOUGHT);
                    }
                } else {
                    (void) admit_runtime_emit_text(
                            ctx,
                            planner_reasoning,
                            loop_origin,
                            LLAMA_COG_LOOP_PHASE_PROPOSE,
                            source_id,
                            plan_id,
                            0,
                            LLAMA_SELF_COG_ARTIFACT_HIDDEN_THOUGHT);
                }
                (void) llama_shared_cognitive_context_get_window(ctx, &context_window);
            }

            if (resumed_task.react_origin == SERVER_REACT_ORIGIN_ACTIVE) {
                if (react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT &&
                        resumed_task.react_resuming_from_tool_result) {
                    if (resumed_task.react_retry_count < resumed_task.react_retry_limit) {
                        resumed_task.react_retry_count += 1;
                        resumed_task.react_retry_feedback =
                                "The previous control step chose wait immediately after a completed tool observation. "
                                "Based on that observation, choose answer, ask, or continue the staged tool protocol. Only choose wait if another external tool is still pending.";
                        if (prepare_react_prompt(resumed_task)) {
                            queue_tasks.post(std::move(resumed_task), true);
                            return;
                        }
                    }
                    expire_foreground_consent_scope(resumed_task.active_trace.episode_id);
                    send_error(slot, "Authoritative ReAct chose wait after a completed tool observation without scheduling further work", ERROR_TYPE_SERVER);
                    return;
                }

                switch (react_step.action) {
                    case LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER:
                        resumed_task.active_trace.winner_action = LLAMA_ACTIVE_LOOP_ACTION_ANSWER;
                        (void) llama_cognitive_active_authoritative_finish(
                                ctx,
                                resumed_task.active_trace.episode_id,
                                LLAMA_COG_TERMINAL_ANSWER_READY);
                        break;
                    case LLAMA_AUTHORITATIVE_REACT_ACTION_ASK:
                        resumed_task.active_trace.winner_action = LLAMA_ACTIVE_LOOP_ACTION_ASK;
                        (void) llama_cognitive_active_authoritative_finish(
                                ctx,
                                resumed_task.active_trace.episode_id,
                                LLAMA_COG_TERMINAL_ASK_USER);
                        break;
                    case LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT:
                        resumed_task.active_trace.winner_action = LLAMA_ACTIVE_LOOP_ACTION_WAIT;
                        (void) llama_cognitive_active_authoritative_finish(
                                ctx,
                                resumed_task.active_trace.episode_id,
                                LLAMA_COG_TERMINAL_WAITING_ON_TOOL);
                        break;
                    default:
                        resumed_task.active_trace.winner_action = LLAMA_ACTIVE_LOOP_ACTION_ACT;
                        break;
                }
                if (!planner_reasoning.empty()) {
                    const std::vector<llama_token> reasoning_tokens =
                            common_tokenize(vocab, planner_reasoning, true, true);
                    if (!reasoning_tokens.empty()) {
                        (void) llama_cognitive_active_planner_reasoning_note(
                                ctx,
                                resumed_task.active_trace.episode_id,
                                reasoning_tokens.data(),
                                reasoning_tokens.size());
                    }
                }
                resumed_task.active_trace.authoritative_turn.valid = true;
                resumed_task.active_trace.authoritative_turn.action = react_step.action;
                resumed_task.active_trace.authoritative_turn.thought_context_item_id = context_window.newest_item_id;
                std::snprintf(
                        resumed_task.active_trace.authoritative_turn.tool_call_payload,
                        sizeof(resumed_task.active_trace.authoritative_turn.tool_call_payload),
                        "%s",
                        tool_xml_payload.c_str());
                resumed_task.active_trace.authoritative_turn.status =
                        react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ACT ?
                                LLAMA_AUTHORITATIVE_TURN_STATUS_WAITING_ON_TOOL :
                                LLAMA_AUTHORITATIVE_TURN_STATUS_COMPLETED;
                capture_active_loop_provenance("active_final", resumed_task.active_trace);

                if (react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ACT) {
                    int32_t pending_command_id = -1;
                    std::string request_error;
                    const bool queued =
                            configure_tool_request_from_chat_call(
                                    resumed_task,
                                    react_step.assistant_msg,
                                    &pending_command_id,
                                    &request_error) &&
                            pending_command_id > 0;
                    if (!queued) {
                        if (pending_command_id > 0) {
                            (void) llama_cognitive_command_complete(ctx, pending_command_id, true);
                        }
                        if (retry_staged_react(resumed_task, "dispatch_rejected", "Tool dispatch rejected: " + request_error)) {
                            return;
                        }
                        expire_foreground_consent_scope(resumed_task.active_trace.episode_id);
                        send_error(resumed_task, "Failed to queue authoritative ReAct tool dispatch: " + request_error, ERROR_TYPE_SERVER);
                        return;
                    }
                    if (!tool_xml_payload.empty()) {
                        (void) admit_runtime_emit_text(
                                ctx,
                                tool_xml_payload,
                                LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                                LLAMA_COG_LOOP_PHASE_PREPARE_TOOL,
                                pending_command_id,
                                resumed_task.active_trace.plan.plan_id,
                                0,
                                LLAMA_SELF_COG_ARTIFACT_TOOL_CALL);
                        (void) llama_shared_cognitive_context_get_window(ctx, &context_window);
                        resumed_task.active_trace.authoritative_turn.tool_call_context_item_id = context_window.newest_item_id;
                    }
                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        resumed_task.react_resuming_from_tool_result = false;
                        resumed_task.react_stage_retry_count = 0;
                        resumed_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL;
                        resumed_task.react_last_tool_family_id = resumed_task.react_selected_tool_family_id;
                        resumed_task.react_last_tool_method_name = resumed_task.react_selected_method_name;
                        resumed_task.react_last_tool_capability_id = resumed_task.react_selected_capability_id;
                        record_react_last_step(
                                resumed_task,
                                "emit_arguments",
                                react_step.tool_xml,
                                react_json_text({
                                        {"tool_family_id", resumed_task.react_last_tool_family_id},
                                        {"method_name", resumed_task.react_last_tool_method_name},
                                        {"capability_id", resumed_task.react_last_tool_capability_id},
                                }));
                        append_react_history_entry(
                                resumed_task,
                                "emit_arguments",
                                "tool_call",
                                react_json_text({
                                        {"tool_family_id", resumed_task.react_last_tool_family_id},
                                        {"method_name", resumed_task.react_last_tool_method_name},
                                        {"capability_id", resumed_task.react_last_tool_capability_id},
                                        {"arguments_json", react_step.tool_xml},
                                }));
                        resumed_task.react_selected_tool_family_id.clear();
                        resumed_task.react_selected_method_name.clear();
                        resumed_task.react_selected_capability_id.clear();
                        resumed_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                        waiting_active_tasks[pending_command_id] = std::move(resumed_task);
                    }
                    const bool dispatched = dispatch_pending_tool_commands(LLAMA_COG_COMMAND_ORIGIN_ACTIVE);
                    if (dispatched) {
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        waiting_active_tasks.erase(pending_command_id);
                    }
                    expire_foreground_consent_scope(resumed_task.active_trace.episode_id);
                    send_error(slot, "Failed to dispatch authoritative ReAct tool command", ERROR_TYPE_SERVER);
                    return;
                }

                if (resumed_task.react_resuming_from_tool_result &&
                        response_text_claims_external_inaccessibility(react_step.assistant_msg.content)) {
                    if (retry_staged_react(
                                resumed_task,
                                "terminal_refusal_conflicts_with_tool_use",
                                "The previous response incorrectly claimed that external tool access was unavailable even though a tool invocation was just attempted. Summarize the observed tool result or continue the staged tool protocol instead.")) {
                        return;
                    }
                    expire_foreground_consent_scope(resumed_task.active_trace.episode_id);
                    send_error(
                            resumed_task,
                            "Authoritative ReAct produced a terminal response that contradicted the observed tool invocation",
                            ERROR_TYPE_SERVER);
                    return;
                }

                const std::string visible_reply = trim_ascii_copy(react_step.assistant_msg.content);
                if (!visible_reply.empty()) {
                    append_react_history_entry(
                            resumed_task,
                            "emit_response",
                            "terminal_reply",
                            visible_reply);
                    (void) admit_runtime_emit_text(
                            ctx,
                            visible_reply,
                            LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                            LLAMA_COG_LOOP_PHASE_FINISH,
                            resumed_task.active_trace.episode_id,
                            resumed_task.active_trace.plan.plan_id,
                            0,
                            LLAMA_SELF_COG_ARTIFACT_VISIBLE_OUTPUT);
                    (void) llama_shared_cognitive_context_get_window(ctx, &context_window);
                    resumed_task.active_trace.authoritative_turn.visible_output_context_item_id = context_window.newest_item_id;
                    const bool published_deferred_message =
                            publish_terminal_deferred_telegram_delivery(
                                    resumed_task,
                                    visible_reply,
                                    "telegram_relay_active",
                                    "final");
                    if (!published_deferred_message && resumed_task.telegram_dialogue_active) {
                        append_telegram_dialogue_assistant_turn(
                                resumed_task.telegram_chat_scope,
                                visible_reply,
                                "telegram_relay_active");
                        mark_runtime_state_dirty("telegram-dialogue-active-output");
                    }
                    if (published_deferred_message) {
                        react_step.assistant_msg.content.clear();
                        slot.n_sent_text = 0;
                    }
                } else {
                    const bool published_fallback = publish_terminal_deferred_telegram_delivery(
                            resumed_task,
                            visible_reply,
                            "telegram_relay_active",
                            "final");
                    if (published_fallback) {
                        append_react_history_entry(
                                resumed_task,
                                "emit_response",
                                "terminal_reply",
                                synthesize_deferred_terminal_failure_text(resumed_task.foreground_text));
                        slot.n_sent_text = 0;
                    }
                }
                slot.generated_text = react_step.assistant_msg.content;
                expire_foreground_consent_scope(resumed_task.active_trace.episode_id);
            } else {
                resumed_task.dmn_trace.authoritative_turn.valid = true;
                resumed_task.dmn_trace.authoritative_turn.action = react_step.action;
                resumed_task.dmn_trace.authoritative_turn.thought_context_item_id = context_window.newest_item_id;
                std::snprintf(
                        resumed_task.dmn_trace.authoritative_turn.tool_call_payload,
                        sizeof(resumed_task.dmn_trace.authoritative_turn.tool_call_payload),
                        "%s",
                        tool_xml_payload.c_str());
                if (react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_ACT) {
                    int32_t pending_command_id = -1;
                    std::string request_error;
                    const bool queued =
                            configure_tool_request_from_chat_call(
                                    resumed_task,
                                    react_step.assistant_msg,
                                    &pending_command_id,
                                    &request_error) &&
                            pending_command_id > 0;
                    if (!queued) {
                        if (retry_staged_react(resumed_task, "dispatch_rejected", "Tool dispatch rejected: " + request_error)) {
                            return;
                        }
                        SRV_WRN("failed to queue authoritative DMN tool dispatch: %s\n", request_error.c_str());
                        (void) llama_cognitive_dmn_authoritative_finish(
                                ctx,
                                resumed_task.dmn_trace.tick_id,
                                LLAMA_COG_TERMINAL_GOVERNANCE_BLOCKED);
                        return;
                    }
                    if (!tool_xml_payload.empty()) {
                        (void) admit_runtime_emit_text(
                                ctx,
                                tool_xml_payload,
                                LLAMA_COG_COMMAND_ORIGIN_DMN,
                                LLAMA_COG_LOOP_PHASE_PREPARE_TOOL,
                                pending_command_id,
                                resumed_task.dmn_trace.plan.plan_id,
                                0,
                                LLAMA_SELF_COG_ARTIFACT_TOOL_CALL);
                        (void) llama_shared_cognitive_context_get_window(ctx, &context_window);
                        resumed_task.dmn_trace.authoritative_turn.tool_call_context_item_id = context_window.newest_item_id;
                    }
                    resumed_task.dmn_trace.winner_action = LLAMA_DMN_ACTION_INVOKE_TOOL;
                    resumed_task.dmn_trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_WAITING_ON_TOOL;
                    std::snprintf(
                            resumed_task.dmn_trace.reasoning_text,
                            sizeof(resumed_task.dmn_trace.reasoning_text),
                            "%s",
                            planner_reasoning.c_str());
                    capture_dmn_provenance("dmn_authoritative_final", resumed_task.dmn_trace);
                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        resumed_task.react_resuming_from_tool_result = false;
                        resumed_task.react_stage_retry_count = 0;
                        resumed_task.react_tool_stage = SERVER_REACT_TOOL_STAGE_DECIDE_AFTER_TOOL;
                        resumed_task.react_last_tool_family_id = resumed_task.react_selected_tool_family_id;
                        resumed_task.react_last_tool_method_name = resumed_task.react_selected_method_name;
                        resumed_task.react_last_tool_capability_id = resumed_task.react_selected_capability_id;
                        record_react_last_step(
                                resumed_task,
                                "emit_arguments",
                                react_step.tool_xml,
                                react_json_text({
                                        {"tool_family_id", resumed_task.react_last_tool_family_id},
                                        {"method_name", resumed_task.react_last_tool_method_name},
                                        {"capability_id", resumed_task.react_last_tool_capability_id},
                                }));
                        resumed_task.react_selected_tool_family_id.clear();
                        resumed_task.react_selected_method_name.clear();
                        resumed_task.react_selected_capability_id.clear();
                        resumed_task.react_pending_terminal_action = LLAMA_AUTHORITATIVE_REACT_ACTION_NONE;
                        waiting_dmn_tasks[pending_command_id] = std::move(resumed_task);
                    }
                    if (dispatch_pending_tool_commands(LLAMA_COG_COMMAND_ORIGIN_DMN)) {
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        waiting_dmn_tasks.erase(pending_command_id);
                    }
                    SRV_WRN("%s\n", "failed to dispatch authoritative DMN tool command");
                    return;
                }

                if (react_step.action == LLAMA_AUTHORITATIVE_REACT_ACTION_INTERNAL_WRITE) {
                    (void) admit_runtime_emit_text(
                            ctx,
                            react_step.assistant_msg.content,
                            LLAMA_COG_COMMAND_ORIGIN_DMN,
                            LLAMA_COG_LOOP_PHASE_OBSERVE,
                            resumed_task.dmn_trace.tick_id,
                            resumed_task.dmn_trace.plan.plan_id,
                            0,
                            LLAMA_SELF_COG_ARTIFACT_INTERNAL_SUMMARY);
                    (void) llama_shared_cognitive_context_get_window(ctx, &context_window);
                    resumed_task.dmn_trace.authoritative_turn.visible_output_context_item_id = context_window.newest_item_id;
                    resumed_task.dmn_trace.winner_action = LLAMA_DMN_ACTION_INTERNAL_WRITE;
                    resumed_task.dmn_trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_COMPLETED;
                    std::snprintf(
                            resumed_task.dmn_trace.reasoning_text,
                            sizeof(resumed_task.dmn_trace.reasoning_text),
                            "%s",
                            planner_reasoning.c_str());
                    capture_dmn_provenance("dmn_authoritative_final", resumed_task.dmn_trace);
                    (void) llama_cognitive_dmn_authoritative_finish(
                            ctx,
                            resumed_task.dmn_trace.tick_id,
                            LLAMA_COG_TERMINAL_INTERNAL_WRITE_READY);
                    return;
                }

                resumed_task.dmn_trace.winner_action = LLAMA_DMN_ACTION_SILENT;
                resumed_task.dmn_trace.authoritative_turn.status = LLAMA_AUTHORITATIVE_TURN_STATUS_COMPLETED;
                std::snprintf(
                        resumed_task.dmn_trace.reasoning_text,
                        sizeof(resumed_task.dmn_trace.reasoning_text),
                        "%s",
                        planner_reasoning.c_str());
                capture_dmn_provenance("dmn_authoritative_final", resumed_task.dmn_trace);
                (void) llama_cognitive_dmn_authoritative_finish(
                        ctx,
                        resumed_task.dmn_trace.tick_id,
                        LLAMA_COG_TERMINAL_PRESSURE_NOT_ADMITTED);
                return;
            }
        }

        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx, true);
        res->response_fields = slot.task->params.response_fields;

        res->truncated           = slot.truncated;
        res->n_decoded           = slot.n_decoded;
        res->n_prompt_tokens     = slot.task->n_tokens();
        res->n_tokens_cached     = slot.prompt.n_tokens();
        res->has_new_line        = slot.has_new_line;
        res->stopping_word       = slot.stopping_word;
        res->stop                = slot.stop;
        res->post_sampling_probs = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters
        if (parsed_react &&
                slot.task &&
                slot.task->react_origin == SERVER_REACT_ORIGIN_ACTIVE) {
            res->has_precomputed_oaicompat_msg = true;
            res->oaicompat_msg = react_step.assistant_msg;
        }

        if (slot.n_sent_text > 0) {
            const std::string & emitted_buffer = !res->content.empty() ? res->content : slot.generated_text;
            const std::string emitted_text = emitted_buffer.substr(0, std::min(slot.n_sent_text, emitted_buffer.size()));
            if (!admit_runtime_emit_text(
                        ctx,
                        emitted_text,
                        LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                        LLAMA_FUNCTIONAL_MICROPHASE_NONE,
                        slot.id,
                        slot.task && slot.task->has_active_trace ? slot.task->active_trace.plan.plan_id : -1)) {
                SLT_WRN(slot, "%s\n", "failed to admit emitted assistant text into self-state");
            }
        }

        if (slot.task->has_active_trace && slot.n_sent_text > 0) {
            (void) llama_active_loop_note_emit(ctx, slot.task->active_trace.episode_id, slot.n_sent_text);
            mark_runtime_state_dirty("active-loop-emit");
        } else if (slot.n_sent_text > 0) {
            mark_runtime_state_dirty("foreground-emit");
        }

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(ctx, i);
            } else {
                embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            apply_core_system_prompt_prefix(task);
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    if (active_loop_enabled &&
                            authoritative_react_control_enabled &&
                            !task.is_child() &&
                            server_task_should_prepare_active_authoritative(task)) {
                        const auto loop_tokens = make_self_state_token_view(
                                task.tokens,
                                &task.active_loop_tokens,
                                core_system_prompt_prefix_tokens.size());
                        const llama_self_state_event loop_event = {
                            /*.tokens =*/ loop_tokens.data,
                            /*.n_tokens =*/ loop_tokens.size,
                            /*.role =*/ task.foreground_role,
                                /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
                                /*.flags =*/ task.foreground_flags,
                                /*.decoder_entropy =*/ 0.0f,
                                /*.decoder_top_margin =*/ 1.0f,
                        };
                        task.has_active_trace =
                                llama_cognitive_active_authoritative_prepare(ctx, &loop_event, &task.active_trace) == 0;
                        if (task.has_active_trace) {
                            if (authoritative_react_control_enabled) {
                                task.react_origin = SERVER_REACT_ORIGIN_ACTIVE;
                            }
                            register_foreground_consent_scope(task);

                            const bool use_react_step =
                                    authoritative_react_control_enabled &&
                                    server_task_should_prepare_authoritative_react(task);
                            if (use_react_step) {
                                if (!prepare_react_prompt(task)) {
                                    send_error(task, "Failed to prepare authoritative ReAct prompt", ERROR_TYPE_SERVER);
                                    break;
                                }
                            } else {
                                (void) dispatch_pending_tool_commands(LLAMA_COG_COMMAND_ORIGIN_ACTIVE);

                                int32_t pending_command_id = -1;
                                int32_t pending_tool_kind = LLAMA_TOOL_KIND_NONE;
                                if (active_runner_waiting_on_tool(&pending_command_id, &pending_tool_kind)) {
                                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                                    waiting_active_tasks[pending_command_id] = std::move(task);
                                    SRV_INF("parked active task for external tool: command=%d tool=%d waiting=%zu\n",
                                            pending_command_id,
                                            pending_tool_kind,
                                            waiting_active_tasks.size());
                                    break;
                                }
                            }
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1
                                            ? get_slot_by_id(id_slot)
                                            : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;
                    res->n_external_bash_pending = count_pending_external_work(bash_work);
                    res->n_external_hard_memory_pending = count_pending_external_work(hard_memory_work);

                    const vicuna_mailbox_snapshot mailbox_snapshot = proactive_mailbox_snapshot_copy(proactive_mailbox);
                    res->n_proactive_responses = (int) mailbox_snapshot.response_order.size();
                    res->n_proactive_live_events = (int) mailbox_snapshot.live_events.size();
                    res->proactive_live_stream_connected = mailbox_snapshot.live_stream_connected;
                    res->proactive_publish_total = mailbox_snapshot.publish_total;
                    res->proactive_complete_total = mailbox_snapshot.complete_total;
                    res->proactive_fail_total = mailbox_snapshot.fail_total;
                    res->proactive_dropped_total = mailbox_snapshot.dropped_total;
                    res->proactive_last_publish_ms = mailbox_snapshot.last_publish_ms;

                    {
                        std::lock_guard<std::mutex> lock(runtime_state_mutex);
                        res->n_waiting_active_tasks = (int) waiting_active_tasks.size();
                        res->runtime_persistence_enabled = runtime_persistence.enabled;
                        res->runtime_persistence_healthy = runtime_persistence.healthy;
                        res->runtime_restore_attempted = runtime_persistence.restore_attempted;
                        res->runtime_restore_success = runtime_persistence.restore_success;
                        res->runtime_last_persist_ms = runtime_persistence.last_persist_ms;
                        res->runtime_last_restore_ms = runtime_persistence.last_restore_ms;
                        res->runtime_persist_success_total = runtime_persistence.persist_success_total;
                        res->runtime_persist_fail_total = runtime_persistence.persist_fail_total;
                        res->n_external_bash_dispatch_total = external_observability.bash_dispatch_total;
                        res->n_external_bash_complete_total = external_observability.bash_complete_total;
                        res->n_external_bash_fail_total = external_observability.bash_fail_total;
                        res->n_external_hard_memory_dispatch_total = external_observability.hard_memory_dispatch_total;
                        res->n_external_hard_memory_complete_total = external_observability.hard_memory_complete_total;
                        res->n_external_hard_memory_fail_total = external_observability.hard_memory_fail_total;
                        res->provenance_enabled = provenance_repository.enabled;
                        res->provenance_healthy = provenance_repository.healthy;
                        res->provenance_last_append_ms = provenance_repository.last_append_ms;
                        res->provenance_append_total = provenance_repository.append_total;
                        res->provenance_append_fail_total = provenance_repository.append_fail_total;
                        res->provenance_active_loop_total = provenance_repository.active_loop_total;
                        res->provenance_tool_result_total = provenance_repository.tool_result_total;
                        res->provenance_dmn_total = provenance_repository.dmn_total;
                        res->provenance_discovered_increase_total = provenance_repository.discovered_increase_total;
                        res->provenance_permanent_increase_total = provenance_repository.permanent_increase_total;
                        res->provenance_allostatic_increase_total = provenance_repository.allostatic_increase_total;
                        res->provenance_functional_update_total = provenance_repository.functional_update_observed_total;
                        res->provenance_process_update_total = provenance_repository.process_update_observed_total;
                        res->provenance_self_state_active_count = provenance_repository.last_snapshot.active_extensions;
                        res->provenance_self_state_discovered_count = provenance_repository.last_snapshot.discovered_extensions;
                        res->provenance_self_state_permanent_count = provenance_repository.last_snapshot.permanent_extensions;
                        res->provenance_self_state_allostatic_count = provenance_repository.last_snapshot.allostatic_extensions;
                        res->provenance_self_state_allostatic_divergence = provenance_repository.last_snapshot.allostatic_divergence;
                        res->provenance_self_state_promotion_readiness = provenance_repository.last_snapshot.promotion_readiness;
                        res->provenance_self_state_belief_pressure = provenance_repository.last_snapshot.belief_pressure;
                    }

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_text_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string;
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    // Legacy dispatcher remains monolithic while server task semantics are being untangled.
    // NOLINTNEXTLINE(readability-function-size)
    void update_slots() {
        (void) llama_past_lora_tick(ctx, ggml_time_us());
        (void) drain_completed_external_work(-1);
        drain_codex_completion_messages();

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

	            if (all_idle) {
                const uint64_t now_us = ggml_time_us();
                bool dmn_due = true;
                const authoritative_react_live_state authoritative_live = collect_authoritative_react_live_state();
                if (ctx) {
                    (void) llama_self_state_refresh_time(ctx);
                }
                llama_cognitive_host_state host_state = {};
                if (llama_cognitive_get_host_state(ctx, &host_state) == 0 &&
                        host_state.last_dmn_time_us > 0 &&
                        now_us > host_state.last_dmn_time_us &&
                        now_us - host_state.last_dmn_time_us < VICUNA_DMN_IDLE_TICK_MIN_INTERVAL_US) {
                    dmn_due = false;
                }
                llama_dmn_tick_trace dmn_trace = {};
                if (dmn_due &&
                        authoritative_live.any_live() &&
                        llama_dmn_defer(ctx, now_us, &dmn_trace) == 0 &&
                        dmn_trace.deferred_for_foreground) {
                    SRV_INF("dmn tick deferred authoritative_loop waiting_active=%d waiting_dmn=%d active_runner=%d dmn_runner=%d\n",
                            authoritative_live.waiting_active_tasks ? 1 : 0,
                            authoritative_live.waiting_dmn_tasks ? 1 : 0,
                            authoritative_live.active_runner_live ? 1 : 0,
                            authoritative_live.dmn_runner_live ? 1 : 0);
                    capture_dmn_provenance("dmn_tick_deferred_authoritative_loop", dmn_trace);
                    mark_runtime_state_dirty("dmn-deferred-authoritative-loop");
                } else if (dmn_due && llama_dmn_tick(ctx, now_us, &dmn_trace) == 0 && dmn_trace.admitted) {
	                    SRV_INF("dmn tick %d winner=%d score=%.3f burst=%d\n",
	                            dmn_trace.tick_id,
	                            dmn_trace.winner_action,
	                            dmn_trace.winner_score,
	                            dmn_trace.burst_count);
	                    capture_dmn_provenance("dmn_tick", dmn_trace);
	                    llama_cognitive_dmn_runner_status runner = {};
                    llama_cognitive_command command = {};
                    if (llama_cognitive_dmn_runner_get(ctx, &runner) == 0 && runner.pending_command_id > 0) {
                        for (int32_t i = 0, n = llama_cognitive_command_count(ctx); i < n; ++i) {
                            if (llama_cognitive_command_get(ctx, i, &command) == 0 &&
                                    command.command_id == runner.pending_command_id) {
                                SRV_INF("dmn runner tick %d command=%d kind=%d status=%d\n",
                                        runner.tick_id,
                                        command.command_id,
                                        command.kind,
                                        command.status);
                                break;
                            }
                        }
                    } else if (authoritative_react_control_enabled &&
                            runner.active &&
                            !runner.waiting_on_tool &&
                            runner.pending_command_id <= 0) {
                        if (!enqueue_dmn_react_task(dmn_trace)) {
                            capture_dmn_provenance("dmn_react_enqueue_failed", dmn_trace);
                            SRV_WRN("failed to queue authoritative DMN ReAct task for tick %d\n", dmn_trace.tick_id);
                        } else {
                            mark_runtime_state_dirty("dmn-react-task");
                            return;
                        }
                    }
                    mark_runtime_state_dirty("dmn-tick");
                }
                (void) dispatch_pending_self_emit_commands(-1);
                (void) dispatch_pending_tool_commands(-1);
                (void) drain_completed_external_work(-1);
                drain_codex_completion_messages();
                reload_openclaw_catalog_if_needed();
                if (!has_waiting_active_tasks() &&
                    count_pending_external_work(bash_work) == 0 &&
                    count_pending_external_work(hard_memory_work) == 0 &&
                    count_pending_external_work(codex_work) == 0) {
                    (void) persist_runtime_state("idle");
                }
                size_t waiting_active_count = 0;
                {
                    std::lock_guard<std::mutex> lock(runtime_state_mutex);
                    waiting_active_count = waiting_active_tasks.size();
                }
                const vicuna_telegram_outbox_snapshot outbox_snapshot = telegram_outbox_snapshot_copy(telegram_outbox);
                SRV_INF("all slots are idle waiting_active=%zu bash_pending=%d hard_memory_pending=%d codex_pending=%d telegram_outbox_items=%zu\n",
                        waiting_active_count,
                        count_pending_external_work(bash_work),
                        count_pending_external_work(hard_memory_work),
                        count_pending_external_work(codex_work),
                        outbox_snapshot.items.size());

                return;
            }
        }

        (void) llama_dmn_defer(ctx, ggml_time_us(), nullptr);

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                llama_tokens evicted_tokens;
                llama_tokens compacted_tokens;
                {
                    const llama_tokens & prompt_tokens = slot.prompt.tokens.get_text_tokens();
                    if (n_discard > 0 && prompt_tokens.size() >= static_cast<size_t>(n_keep + n_discard)) {
                        evicted_tokens.assign(
                                prompt_tokens.begin() + n_keep,
                                prompt_tokens.begin() + n_keep + n_discard);
                    }

                    compacted_tokens = prompt_tokens;
                    for (size_t i = n_keep + n_discard; i < compacted_tokens.size(); i++) {
                        compacted_tokens[i - n_discard] = compacted_tokens[i];
                    }
                    compacted_tokens.resize(slot.prompt.tokens.size() - n_discard);
                }

                llama_active_lora_stats active_before = {};
                bool have_active_before = llama_active_lora_get_stats(ctx, &active_before) == 0;
                bool active_weights_changed = false;

                if (!evicted_tokens.empty()) {
                    llama_self_state_event evicted_event = {
                        /*.tokens =*/ evicted_tokens.data(),
                        /*.n_tokens =*/ evicted_tokens.size(),
                        /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
                        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
                        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED |
                                LLAMA_SELF_STATE_EVENT_INTERNAL_ARTIFACT |
                                LLAMA_SELF_STATE_EVENT_CONTEXT_COMPACTED,
                        /*.decoder_entropy =*/ 0.0f,
                        /*.decoder_top_margin =*/ 1.0f,
                        /*.artifact_kind =*/ LLAMA_SELF_COG_ARTIFACT_CONTEXT_EVICTION,
                        /*.loop_origin =*/ LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                        /*.phase =*/ LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_COMPRESSION,
                        /*.source_id =*/ slot.id,
                        /*.plan_id =*/ -1,
                    };
                    llama_self_state_feature_vector evicted_pre = {};
                    llama_self_state_feature_vector evicted_post = {};
                    const bool have_self_state_event =
                            llama_self_state_build_prewrite_features(ctx, &evicted_event, &evicted_pre) == 0 &&
                            llama_self_state_apply_prewrite(ctx, &evicted_event, &evicted_pre) == 0 &&
                            llama_self_state_build_postwrite_features(ctx, &evicted_event, &evicted_post) == 0 &&
                            llama_self_state_apply_postwrite(ctx, &evicted_event, &evicted_post) == 0;
                    if ((have_self_state_event &&
                            llama_active_lora_ingest_event(ctx, &evicted_event, &evicted_post) != 0) ||
                        (!have_self_state_event &&
                            llama_active_lora_ingest(ctx, evicted_tokens.data(), evicted_tokens.size()) != 0)) {
                        SLT_WRN(slot, "%s\n", "Active LoRA ingestion failed for evicted span");
                    } else if (have_active_before) {
                        llama_active_lora_stats active_after = {};
                        if (llama_active_lora_get_stats(ctx, &active_after) == 0) {
                            active_weights_changed = active_after.updates_applied > active_before.updates_applied;
                        }
                    }
                }

                const bool schedule_strict_replay = active_weights_changed && !compacted_tokens.empty();

                if (schedule_strict_replay) {
                    SLT_WRN(slot, "strict KV replay scheduled after Active LoRA update, replay_tokens = %zu\n", compacted_tokens.size());

                    llama_memory_seq_rm(llama_get_memory(ctx), slot.id, -1, -1);
                    slot.prompt.tokens.clear();
                    slot.prompt.checkpoints.clear();
                    slot.replay_tokens = std::make_unique<server_tokens>(compacted_tokens, false);
                    slot.state = SLOT_STATE_REPLAYING_PROMPT;
                    slot.i_batch = -1;
                    slot.i_batch_dft.clear();
                    slot.drafted.clear();
                    slot.n_prompt_tokens_cache = 0;
                    slot.n_prompt_tokens_processed = 0;

                    if (slot.can_speculate()) {
                        common_speculative_begin(slot.spec, compacted_tokens);
                    }
                } else {
                    llama_memory_seq_rm (llama_get_memory(ctx), slot.id, n_keep            , n_keep + n_discard);
                    llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);
                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(compacted_tokens);

                    if (active_weights_changed && compacted_tokens.empty()) {
                        SLT_WRN(slot, "%s\n", "strict KV replay skipped because no retained tokens remain");
                    } else if (!active_weights_changed) {
                        SLT_DBG(slot, "%s\n", "strict KV replay skipped because Active LoRA weights did not change");
                    }
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // first, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            // generate draft tokens in speculative decoding mode
            // TODO: rework to have a single draft llama_context shared across all slots [TAG_SERVER_SPEC_REWORK]
            //       perform the speculative drafting for all sequences at the same time in a single batch
            const int n_draft_max = slot.get_n_draft_max();
            if (n_draft_max > 0) {
                if (mctx) {
                    // we should never reach this, as speculative is automatically disabled if mmproj is loaded
                    GGML_ABORT("not supported by multimodal");
                }

                const llama_tokens & cached_text_tokens = slot.prompt.tokens.get_text_tokens();

                const auto & params_spec = slot.task->params.speculative;

                llama_tokens draft = common_speculative_draft(slot.spec, params_spec, cached_text_tokens, slot.sampled);

                if (draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(slot, "draft size %d exceeds max %d, truncating\n", (int) draft.size(), n_draft_max);
                    draft.resize(n_draft_max);
                }

                // add the sampled token to the batch
                slot.i_batch_dft.push_back(batch.n_tokens);
                common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);
                slot.prompt.tokens.push_back(slot.sampled);

                if (slot.task->params.speculative.n_min > (int) draft.size()) {
                    SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) draft.size(), slot.task->params.speculative.n_min);
                    // fallback to normal decoding
                    slot.i_batch = slot.i_batch_dft[0];
                    slot.drafted.clear();
                    slot.i_batch_dft.clear();
                } else {
                    // keep track of total number of drafted tokens tested
                    slot.n_draft_total += draft.size();

                    // add all drafted tokens to the batch
                    for (size_t i = 0; i < draft.size(); i++) {
                        slot.i_batch_dft.push_back(batch.n_tokens);
                        common_batch_add(batch, draft[i], slot.prompt.tokens.pos_next(), { slot.id }, true);
                        slot.prompt.tokens.push_back(draft[i]);
                    }
                    slot.drafted = std::move(draft);
                }
            } else {
                // no speculative decoding
                slot.i_batch = batch.n_tokens;

                common_batch_add(batch, slot.sampled, slot.prompt.tokens.pos_next(), { slot.id }, true);

                slot.prompt.tokens.push_back(slot.sampled);

                SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                        slot.n_ctx, slot.prompt.n_tokens(), slot.truncated);
            }
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_REPLAYING_PROMPT) {
                    const bool is_replay = slot.is_replaying_prompt();
                    const auto & input_tokens = slot.input_tokens_ref();
                    const int n_input_tokens = slot.input_tokens_count();

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED || slot.state == SLOT_STATE_REPLAYING_PROMPT) {
                        int n_past = 0;
                        if (slot.state == SLOT_STATE_STARTED) {
                            slot.t_start_process_prompt = ggml_time_us();
                            slot.t_start_generation = 0;
                        }
                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        if (is_replay) {
                            SLT_INF(slot, "strict KV replay started, n_ctx_slot = %d, replay_tokens = %d\n",
                                    slot.n_ctx, n_input_tokens);
                        } else {
                            SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                    slot.n_ctx, slot.task->params.n_keep, n_input_tokens);
                        }

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        }*/

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (n_input_tokens > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               n_input_tokens, n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (n_input_tokens > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        n_input_tokens, slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (n_input_tokens >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         n_input_tokens, slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (!is_replay && slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            llama_memory_seq_rm (llama_get_memory(ctx), slot.id, head_p, head_c);
                                            llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // note: when n_swa == 0, the model does not use SWA, which is equivalent to a window of 1
                            const auto n_swa = std::max(1, llama_model_n_swa(model));

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa);

                            if (n_past > 0 && n_past < slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), input_tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = input_tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min > pos_min_thold) {
                                    SLT_WRN(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d, n_swa = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min, n_swa);

                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = "update_slots"](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            return cur.pos_min < pos_min_thold;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        const size_t checkpoint_size = it->data.size();
                                        const size_t n = llama_state_seq_set_data_ext(ctx, it->data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                            //printf("[DEBUG] `do_reset` was set to `true` after failing to restore a checkpoint");
                                        } else {
                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) checkpoint_size / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_min > pos_min_thold
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_min > pos_min_thold) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, (float) cur.data.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == n_input_tokens && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, n_input_tokens);
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // send initial 0% progress update if needed
                        // this is to signal the client that the request has started processing
                        if (!is_replay && slot.task->params.stream && slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + n_input_tokens > n_batch) {
                            continue;
                        }
                    }

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_INF(slot, "n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, p0, -1)) {
                        SLT_WRN(slot, "failed to truncate tokens with position >= %d - clearing the memory\n", p0);

                        slot.prompt_clear(true);

                        // there is no common part left
                        slot.n_prompt_tokens_cache = 0;
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // check if we should process the image
                    if (!is_replay && slot.prompt.n_tokens() < n_input_tokens && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        do_checkpoint = false; // do not checkpoint right after an image chunk
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && !is_replay && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model uses SWA and we are not using `swa_full`
                    // - the model architecture is marked as recurrent or hybrid
                    //
                    // TODO: try to make this conditional on the context or the memory module, instead of the model type
                    do_checkpoint = do_checkpoint && (
                            llama_model_is_recurrent(model) ||
                            llama_model_is_hybrid(model) ||
                            (llama_model_n_swa(model) > 0 && !params_base.swa_full)
                            );

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < n_input_tokens && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.task->need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (do_checkpoint && n_input_tokens == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == n_input_tokens) {
                        GGML_ASSERT(batch.n_tokens > 0);
                        batch.logits[batch.n_tokens - 1] = true;
                        slot.i_batch = batch.n_tokens - 1;

                        if (is_replay) {
                            slot.init_sampler();
                            slot.replay_tokens.reset();
                            slot.state = SLOT_STATE_GENERATING;
                            SLT_INF(slot, "strict KV replay done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.n_tokens);
                        } else {
                            slot.state = SLOT_STATE_DONE_PROMPT;
                            slot.n_decoded = 0;
                            slot.init_sampler();
                            SLT_INF(slot, "prompt processing done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.n_tokens);
                        }
                    } else {
                        if (n_input_tokens < slot.prompt.n_tokens() + n_ubatch) {
                            // near the end of the prompt
                            do_checkpoint = do_checkpoint && true;
                        } else {
                            // only do non-end checkpoints if the "checkpoint every n tokens" option is set
                            do_checkpoint = do_checkpoint && params_base.checkpoint_every_nt > 0;

                            if (do_checkpoint) {
                                llama_pos last_checkpoint = 0;
                                if (!slot.prompt.checkpoints.empty()) {
                                    last_checkpoint = slot.prompt.checkpoints.back().n_tokens;
                                }

                                do_checkpoint = do_checkpoint && slot.prompt.n_tokens() - batch.n_tokens - last_checkpoint >= params_base.checkpoint_every_nt;

                                if (do_checkpoint) {
                                    SLT_INF(slot, "%d tokens since last checkpoint at %d, creating new checkpoint during processing at position %d\n", params_base.checkpoint_every_nt, last_checkpoint, slot.prompt.n_tokens());
                                }
                            }
                        }

                        SLT_INF(slot, "prompt processing progress, n_tokens = %d, batch.n_tokens = %d, progress = %f\n", slot.prompt.n_tokens(), batch.n_tokens, (float) slot.prompt.n_tokens() / std::max(1, n_input_tokens));
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);

                    // no need for empty or small checkpoints
                    do_checkpoint = do_checkpoint && (pos_min >= 0 && pos_max >= 64);

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || pos_max > slot.prompt.checkpoints.back().pos_max + 64);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
                            // make room for the new checkpoint, if needed
                            const auto & cur = slot.prompt.checkpoints.front();

                            SLT_WRN(slot,
                                    "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                    ", size = %.3f MiB)\n",
                                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);

                            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
                        }

                        const size_t checkpoint_size =
                            llama_state_seq_get_size_ext(ctx, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        auto & cur = slot.prompt.checkpoints.emplace_back(server_prompt_checkpoint{
                            /*.pos_min  = */ pos_min,
                            /*.pos_max  = */ pos_max,
                            /*.n_tokens = */ slot.prompt.n_tokens() - n_tokens_cur,
                            /*.data     = */ std::vector<uint8_t>(checkpoint_size),
                        });

                        llama_state_seq_get_data_ext(ctx, cur.data.data(), checkpoint_size, slot.id,
                                                     LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        SLT_WRN(slot,
                                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                ", size = %.3f MiB)\n",
                                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                                cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        if (slot_batched) {
            // Treat adapter changes as a hard boundary: finish any prior backend
            // work, apply the slot's request/runtime stack, then decode.
            llama_synchronize(ctx);
            common_set_adapter_lora(ctx, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx, slot_batched->task->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(slot.spec, slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (!slot.i_batch_dft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), ctx, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || slot.i_batch_dft.empty()) {
                    continue;
                }

                const size_t n_draft = slot.drafted.size();

                // the accepted tokens from the speculation
                const auto ids = common_sampler_sample_and_accept_n(slot.smpl.get(), ctx, slot.i_batch_dft, slot.drafted);
                slot.i_batch_dft.clear();
                slot.drafted.clear();

                const int64_t t_current = ggml_time_us();

                slot.n_decoded += ids.size();

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                // inform the speculative decoding about the number of accepted tokens
                common_speculative_accept(slot.spec, ids.size() - 1);

                // rollback to the state before sampling the draft tokens
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);

                // add accepted tokens to the prompt
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});
                slot.sampled = ids.back(); // last accepted token

                llama_memory_seq_rm(llama_get_memory(ctx), slot.id, slot.prompt.n_tokens(), -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(ctx, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

// NOLINTNEXTLINE(readability-make-member-function-const)
bool server_context::load_model(const common_params & params) {
    return impl->load_model(params);
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ build_info,
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_webui_settings    */ impl->json_webui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model),
        /* model_n_params         */ llama_model_n_params(impl->model),
        /* model_size             */ llama_model_size(impl->model),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};



//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;
        const std::string telegram_chat_scope = trim_ascii_copy(
                request_header_value_ci(req, "X-Vicuna-Telegram-Chat-Id"));
        const bool telegram_deferred_delivery = parse_bool_header_value(
                request_header_value_ci(req, "X-Vicuna-Telegram-Deferred-Delivery"));
        const bool disable_authoritative_react =
                parse_bool_header_value(request_header_value_ci(req, "X-Vicuna-Disable-Authoritative-React")) ||
                request_param_truthy(req.get_param("disable_authoritative_react")) ||
                json_value(data, "disable_authoritative_react", false);
        const int64_t telegram_message_id = parse_int64_header_value(
                request_header_value_ci(req, "X-Vicuna-Telegram-Message-Id"));
        const int32_t telegram_history_turn_limit = std::max(
                0,
                (int32_t) parse_int64_header_value(
                        request_header_value_ci(req, "X-Vicuna-Telegram-History-Turns")));
        const std::vector<common_chat_msg> telegram_transcript_messages =
                telegram_chat_scope.empty() ? std::vector<common_chat_msg>() :
                extract_telegram_transcript_messages(req.body);
        if (!telegram_chat_scope.empty() && !telegram_transcript_messages.empty()) {
            ctx_server.sync_telegram_dialogue_history(
                    telegram_chat_scope,
                    telegram_message_id,
                    telegram_transcript_messages);
            ctx_server.mark_runtime_state_dirty("telegram-dialogue-sync");
        }

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    data);
            ctx_server.apply_core_system_prompt_prefix(task);
            task.foreground_role = classify_foreground_role_for_request(req.body, data);
            const std::string foreground_text = extract_foreground_message_text_for_request(req.body, data);
            task.foreground_text = foreground_text;
            task.disable_authoritative_react = disable_authoritative_react;
            if (!foreground_text.empty()) {
                task.active_loop_tokens = common_tokenize(ctx_server.vocab, foreground_text, true, true);
            }
            if (!telegram_chat_scope.empty()) {
                task.telegram_dialogue_active = true;
                task.telegram_deferred_delivery = telegram_deferred_delivery;
                task.telegram_chat_scope = telegram_chat_scope;
                task.telegram_message_id = telegram_message_id;
                task.telegram_history_turn_limit = telegram_history_turn_limit;
            }
            if (task.foreground_role == LLAMA_SELF_STATE_EVENT_TOOL) {
                task.foreground_flags |= LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED;
            }
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        }
        if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json arr = json::array();
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
            arr.push_back(res->to_json());
        }
        GGML_ASSERT(!arr.empty() && "empty results");
        if (arr.size() == 1) {
            // if single request, return single object instead of array
            res->ok(arr[0]);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
            // if multiple results in OAI format, we need to re-format them
            json & choices = arr[0]["choices"];
            for (size_t i = 1; i < arr.size(); i++) {
                choices.push_back(std::move(arr[i]["choices"][0]));
            }
            res->ok(arr[0]);
        } else {
            // multi-results, non-OAI compat
            res->ok(arr);
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                }
                return format_oai_sse(json {{ "error", res_json }});
            };

            try {
                if (req.should_stop()) {
                    LOG_DBG("srv  %12.*s: %s", 12, "stream_next", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    LOG_DBG("srv  %12.*s: %s", 12, "stream_next", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    LOG_DBG("srv  %12.*s: %s", 12, "stream_next", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    LOG_DBG("srv  %12.*s: %s", 12, "stream_next", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                }
                GGML_ASSERT(
                    dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                    || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                );
                json res_json = result->to_json();
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    output = format_anthropic_sse(res_json);
                } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                    output = format_oai_resp_sse(res_json);
                } else {
                    output = format_oai_sse(res_json);
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok(this->ctx_server.build_health_json());
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }, {
                    {"name",  "external_bash_dispatch_total"},
                    {"help",  "Total number of bash tool commands dispatched to async workers."},
                    {"value",  res_task->n_external_bash_dispatch_total}
            }, {
                    {"name",  "external_bash_complete_total"},
                    {"help",  "Total number of bash tool commands completed."},
                    {"value",  res_task->n_external_bash_complete_total}
            }, {
                    {"name",  "external_bash_fail_total"},
                    {"help",  "Total number of bash tool commands that failed or were rejected."},
                    {"value",  res_task->n_external_bash_fail_total}
            }, {
                    {"name",  "external_hard_memory_dispatch_total"},
                    {"help",  "Total number of hard-memory queries dispatched to async workers."},
                    {"value",  res_task->n_external_hard_memory_dispatch_total}
            }, {
                    {"name",  "external_hard_memory_complete_total"},
                    {"help",  "Total number of hard-memory queries completed."},
                    {"value",  res_task->n_external_hard_memory_complete_total}
            }, {
                    {"name",  "external_hard_memory_fail_total"},
                    {"help",  "Total number of hard-memory queries that failed."},
                    {"value",  res_task->n_external_hard_memory_fail_total}
            }, {
                    {"name",  "runtime_persist_success_total"},
                    {"help",  "Total number of successful runtime snapshot writes."},
                    {"value",  res_task->runtime_persist_success_total}
            }, {
                    {"name",  "runtime_persist_fail_total"},
                    {"help",  "Total number of failed runtime snapshot writes."},
                    {"value",  res_task->runtime_persist_fail_total}
            }, {
                    {"name",  "proactive_publish_total"},
                    {"help",  "Total number of proactive self-emits published to the OpenAI mailbox."},
                    {"value",  res_task->proactive_publish_total}
            }, {
                    {"name",  "proactive_complete_total"},
                    {"help",  "Total number of proactive self-emits completed and retained."},
                    {"value",  res_task->proactive_complete_total}
            }, {
                    {"name",  "proactive_fail_total"},
                    {"help",  "Total number of proactive self-emits that failed to publish."},
                    {"value",  res_task->proactive_fail_total}
            }, {
                    {"name",  "proactive_dropped_total"},
                    {"help",  "Total number of proactive responses evicted from retention."},
                    {"value",  res_task->proactive_dropped_total}
            }, {
                    {"name",  "provenance_append_total"},
                    {"help",  "Total number of unified provenance events appended."},
                    {"value",  res_task->provenance_append_total}
            }, {
                    {"name",  "provenance_append_fail_total"},
                    {"help",  "Total number of unified provenance append failures."},
                    {"value",  res_task->provenance_append_fail_total}
            }, {
                    {"name",  "provenance_active_loop_total"},
                    {"help",  "Total number of active-loop provenance events."},
                    {"value",  res_task->provenance_active_loop_total}
            }, {
                    {"name",  "provenance_tool_result_total"},
                    {"help",  "Total number of tool-result provenance events."},
                    {"value",  res_task->provenance_tool_result_total}
            }, {
                    {"name",  "provenance_dmn_total"},
                    {"help",  "Total number of admitted DMN provenance events."},
                    {"value",  res_task->provenance_dmn_total}
            }, {
                    {"name",  "provenance_discovered_increase_total"},
                    {"help",  "Observed increases in discovered self-state count across provenance snapshots."},
                    {"value",  res_task->provenance_discovered_increase_total}
            }, {
                    {"name",  "provenance_permanent_increase_total"},
                    {"help",  "Observed increases in permanent self-state count across provenance snapshots."},
                    {"value",  res_task->provenance_permanent_increase_total}
            }, {
                    {"name",  "provenance_allostatic_increase_total"},
                    {"help",  "Observed increases in allostatic self-state count across provenance snapshots."},
                    {"value",  res_task->provenance_allostatic_increase_total}
            }, {
                    {"name",  "provenance_functional_update_total"},
                    {"help",  "Observed functional LoRA update-count increases across provenance snapshots."},
                    {"value",  res_task->provenance_functional_update_total}
            }, {
                    {"name",  "provenance_process_update_total"},
                    {"help",  "Observed process-functional update-count increases across provenance snapshots."},
                    {"value",  res_task->provenance_process_update_total}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  res_task->n_tasks_deferred}
            },{
                    {"name",  "waiting_active_tasks"},
                    {"help",  "Number of active-loop tasks parked awaiting external tool completion."},
                    {"value",  res_task->n_waiting_active_tasks}
            },{
                    {"name",  "external_bash_pending"},
                    {"help",  "Number of pending or running bash tool jobs."},
                    {"value",  res_task->n_external_bash_pending}
            },{
                    {"name",  "external_hard_memory_pending"},
                    {"help",  "Number of pending or running hard-memory jobs."},
                    {"value",  res_task->n_external_hard_memory_pending}
            },{
                    {"name",  "runtime_persistence_enabled"},
                    {"help",  "Whether runtime persistence is enabled."},
                    {"value",  res_task->runtime_persistence_enabled ? 1 : 0}
            },{
                    {"name",  "runtime_persistence_healthy"},
                    {"help",  "Whether runtime persistence is currently healthy."},
                    {"value",  res_task->runtime_persistence_healthy ? 1 : 0}
            },{
                    {"name",  "runtime_restore_success"},
                    {"help",  "Whether the most recent runtime restore succeeded."},
                    {"value",  res_task->runtime_restore_success ? 1 : 0}
            },{
                    {"name",  "provenance_enabled"},
                    {"help",  "Whether the unified provenance repository is enabled."},
                    {"value",  res_task->provenance_enabled ? 1 : 0}
            },{
                    {"name",  "provenance_healthy"},
                    {"help",  "Whether the unified provenance repository is currently healthy."},
                    {"value",  res_task->provenance_healthy ? 1 : 0}
            },{
                    {"name",  "provenance_self_state_active_count"},
                    {"help",  "Latest observed active self-model extension count from provenance snapshots."},
                    {"value",  res_task->provenance_self_state_active_count}
            },{
                    {"name",  "provenance_self_state_discovered_count"},
                    {"help",  "Latest observed discovered self-model extension count from provenance snapshots."},
                    {"value",  res_task->provenance_self_state_discovered_count}
            },{
                    {"name",  "provenance_self_state_permanent_count"},
                    {"help",  "Latest observed permanent self-model extension count from provenance snapshots."},
                    {"value",  res_task->provenance_self_state_permanent_count}
            },{
                    {"name",  "provenance_self_state_allostatic_count"},
                    {"help",  "Latest observed allostatic self-model extension count from provenance snapshots."},
                    {"value",  res_task->provenance_self_state_allostatic_count}
            },{
                    {"name",  "provenance_self_state_allostatic_divergence"},
                    {"help",  "Latest observed allostatic divergence from provenance snapshots."},
                    {"value",  res_task->provenance_self_state_allostatic_divergence}
            },{
                    {"name",  "provenance_self_state_promotion_readiness"},
                    {"help",  "Latest observed promotion readiness from provenance snapshots."},
                    {"value",  res_task->provenance_self_state_promotion_readiness}
            },{
                    {"name",  "provenance_self_state_belief_pressure"},
                    {"help",  "Latest observed residual allostatic pressure from provenance snapshots."},
                    {"value",  res_task->provenance_self_state_belief_pressure}
            },{
                    {"name",  "proactive_responses"},
                    {"help",  "Number of retained proactive responses."},
                    {"value",  res_task->n_proactive_responses}
            },{
                    {"name",  "proactive_live_events"},
                    {"help",  "Number of retained proactive mailbox events."},
                    {"value",  res_task->n_proactive_live_events}
            },{
                    {"name",  "proactive_live_stream_connected"},
                    {"help",  "Whether the single-client proactive live stream is attached."},
                    {"value",  res_task->proactive_live_stream_connected ? 1 : 0}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "api_surface",                 common_server_api_surface_to_str(params.api_surface) },
            { "webui",                       params.webui },
            { "webui_settings",              meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->get_api_show = [this](const server_http_req &) {
        auto res = create_response();
        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        json data = {
            {
                "model_info", {
                    { "llama.context_length", meta->slot_n_ctx },
                }
            },
            {"modelfile", ""},
            {"parameters", ""},
            {"template", tmpl_default},
            {"details", {
                {"parent_model", ""},
                {"format", "gguf"},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
            }},
            {"model_info", ""},
            {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})}
        };

        res->ok(data);
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        LOG_DBG("srv  %12.*s: creating infill tasks, n_prompts = %d\n", 12, "post_infill", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_text_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_responses_to_chatcmpl(json::parse(req.body));
        LOG_DBG("srv  %12.*s: %s\n", 12, "post_responses", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        LOG_DBG("srv  %12.*s: converted request: %s\n", 12, "post_responses", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->get_response_oai = [this](const server_http_req & req) {
        auto res = create_response(true);
        const std::string response_id = req.get_param("response_id");
        if (response_id.empty()) {
            res->error(format_error_response("missing response id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (request_param_truthy(req.get_param("stream"))) {
            std::vector<json> events;
            if (!ctx_server.proactive_mailbox_get_response_events(
                        response_id,
                        parse_uint64_param(req.get_param("after"), 0),
                        &events)) {
                res->status = 404;
                res->error(format_error_response("response not found", ERROR_TYPE_NOT_FOUND));
                return res;
            }
            res->status = 200;
            res->content_type = "text/event-stream";
            res->data = format_oai_resp_sse(events);
            SRV_INF("replayed proactive response stream response=%s events=%zu\n",
                    response_id.c_str(),
                    events.size());
            return res;
        }

        json response;
        if (!ctx_server.proactive_mailbox_get_response(response_id, &response)) {
            res->status = 404;
            res->error(format_error_response("response not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        res->ok(response);
        return res;
    };

    this->get_self_emit_stream_oai = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!ctx_server.proactive_mailbox_connect_live_stream()) {
            json err = format_error_response(
                    "proactive live stream already connected for this instance",
                    ERROR_TYPE_INVALID_REQUEST);
            err["code"] = 409;
            err["type"] = "conflict_error";
            res->error(err);
            return res;
        }

        res->status = 200;
        res->content_type = "text/event-stream";
        uint64_t last_sequence = parse_uint64_param(req.get_param("after"), 0);
        std::vector<json> initial_events = ctx_server.proactive_mailbox_collect_live_events(last_sequence, &last_sequence);
        res->data = format_oai_resp_sse(initial_events);
        res->next = [res_this = res.get(), last_sequence, &req, ctx_this = &this->ctx_server](std::string & output) mutable -> bool {
            if (req.should_stop()) {
                ctx_this->proactive_mailbox_disconnect_live_stream();
                return false;
            }

            if (!res_this->data.empty()) {
                output = std::move(res_this->data);
                res_this->data.clear();
                return true;
            }

            while (!req.should_stop()) {
                std::vector<json> events = ctx_this->proactive_mailbox_collect_live_events(last_sequence, &last_sequence);
                if (!events.empty()) {
                    output = format_oai_resp_sse(events);
                    return true;
                }
                ctx_this->proactive_mailbox_wait_for_live_events(last_sequence, req.should_stop);
            }

            ctx_this->proactive_mailbox_disconnect_live_stream();
            return false;
        };
        return res;
    };

    this->get_telegram_outbox = [this](const server_http_req & req) {
        auto res = create_response(true);
        const uint64_t after_sequence = parse_uint64_param(req.get_param("after"), 0);
        uint64_t last_sequence = after_sequence;
        const std::vector<json> items = ctx_server.telegram_outbox_collect_items(after_sequence, &last_sequence);
        const json health = ctx_server.build_telegram_outbox_health_json();
        res->ok({
            {"items", items},
            {"last_sequence", last_sequence},
            {"stored_items", json_value(health, "stored_items", 0)},
            {"next_sequence_number", json_value(health, "next_sequence_number", uint64_t(1))},
            {"oldest_sequence", json_value(health, "oldest_sequence", uint64_t(0))},
            {"newest_sequence", json_value(health, "newest_sequence", uint64_t(0))},
        });
        return res;
    };

    this->post_telegram_approval = [this](const server_http_req & req) {
        auto res = create_response(true);
        const json body = req.body.empty() ? json::object() : json::parse(req.body);
        auto & ctx_mut = const_cast<server_context_impl &>(ctx_server);
        res->ok(ctx_mut.submit_telegram_approval_decision(body));
        return res;
    };

    this->post_telegram_interruption = [this](const server_http_req & req) {
        auto res = create_response(true);
        const json body = req.body.empty() ? json::object() : json::parse(req.body);
        auto & ctx_mut = const_cast<server_context_impl &>(ctx_server);
        res->ok(ctx_mut.interrupt_pending_dmn_approvals_for_chat(body));
        return res;
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        LOG_DBG("srv  %12.*s: %s\n", 12, "anth_messages", "Request converted: Anthropic -> OpenAI Chat Completions");
        LOG_DBG("srv  %12.*s: converted request: %s\n", 12, "anth_messages", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        LOG_DBG("srv  %12.*s: %s\n", 12, "anth_count", "Request converted: Anthropic -> OpenAI Chat Completions");
        LOG_DBG("srv  %12.*s: converted request: %s\n", 12, "anth_count", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       meta->model_name},
                    {"aliases",  meta->model_aliases},
                    {"tags",     meta->model_tags},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     {
                        {"vocab_type",  meta->model_vocab_type},
                        {"n_vocab",     meta->model_vocab_n_tokens},
                        {"n_ctx_train", meta->model_n_ctx_train},
                        {"n_embd",      meta->model_n_embd_inp},
                        {"n_params",    meta->model_n_params},
                        {"size",        meta->model_size},
                    }},
                },
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        }
        if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    }
    if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    }
    for (auto & res : all_results.results) {
        GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
        responses.push_back(res->to_json());
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}
