#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_active_lora_manager;
class llama_self_state;
class llama_cognitive_loop;
class llama_hard_memory;
class llama_bash_tool;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, const float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    bool active_lora_init(const llama_active_lora_params & params);
    bool active_lora_ingest(const llama_token * tokens, size_t n_tokens);
    bool active_lora_ingest(const llama_self_state_event & event, const llama_self_state_feature_vector * features = nullptr);
    bool active_lora_remediate(const llama_token * tokens, size_t n_tokens, float budget_scale);
    bool active_lora_remediate(const llama_self_state_event & event, float budget_scale, const llama_self_state_feature_vector * features = nullptr);
    bool active_lora_get_stats(llama_active_lora_stats * out_stats) const;
    bool past_lora_init(const llama_past_lora_params & params);
    bool past_lora_tick(uint64_t now_us);
    bool past_lora_get_stats(llama_past_lora_stats * out_stats) const;
    int32_t serving_lora_stack_count() const;
    bool serving_lora_stack_layer(int32_t i, llama_serving_lora_layer_info * out_info) const;
    int32_t functional_lora_family_count() const;
    bool functional_lora_family_config_get(int32_t family, llama_functional_lora_family_config * out_config) const;
    bool functional_lora_family_state_get(int32_t family, llama_functional_lora_family_state * out_state) const;
    bool functional_lora_get_last_trace(llama_functional_lora_trace * out_trace) const;
    bool functional_lora_get_last_update(int32_t family, llama_functional_lora_update_info * out_update) const;
    bool functional_lora_set_ablation(const llama_functional_lora_ablation_config & config);
    bool functional_lora_get_ablation(llama_functional_lora_ablation_config * out_config) const;
    bool functional_lora_activate(const llama_functional_activation_decision & decision);
    bool functional_lora_note_command_complete(int32_t origin);
    bool functional_lora_apply_update(
            int32_t family,
            int32_t loop_origin,
            int32_t start_microphase,
            int32_t settle_microphase,
            const llama_functional_outcome_snapshot & before,
            const llama_functional_outcome_snapshot & after,
            int32_t selected_tool_kind,
            int32_t candidate_count,
            const float * metrics,
            size_t metric_count,
            float signed_outcome,
            float magnitude,
            const llama_self_state_event & event,
            const llama_self_state_feature_vector * features);
    bool self_state_refresh_time();
    bool self_state_set_time(const llama_self_state_time_point & time_point);
    bool self_state_get_datetime(llama_self_state_datetime * out_info) const;
    bool self_state_configure(const llama_self_state_params & params);
    int32_t self_state_register_count() const;
    bool self_state_get_register(int32_t register_id, llama_self_register_info * out_info) const;
    bool self_state_set_channel_state(int32_t channel_state);
    bool self_state_note_user_event();
    bool self_state_note_tool_event();
    bool self_state_note_emit_event();
    bool self_state_set_identity(const llama_token * tokens, size_t n_tokens);
    bool self_state_upsert_goal(int32_t goal_id, const llama_token * tokens, size_t n_tokens, float priority);
    bool self_state_upsert_commitment(int32_t commitment_id, const llama_token * tokens, size_t n_tokens, float priority, bool unresolved);
    int32_t self_state_goal_count() const;
    int32_t self_state_commitment_count() const;
    int32_t self_state_working_memory_count() const;
    bool self_state_upsert_memory_handle(int32_t handle_id, int32_t kind, const llama_token * tokens, size_t n_tokens, float priority);
    int32_t self_state_memory_handle_count() const;
    int32_t self_state_reactivation_count() const;
    bool self_state_get_reactivation(int32_t index, llama_self_reactivation_info * out_info) const;
    bool self_state_upsert_tool_job(int32_t job_id, int32_t status, float importance);
    bool self_state_get_tool_state(llama_self_tool_state_info * out_info) const;
    bool self_state_get_social_state(llama_self_social_state_info * out_info) const;
    bool self_state_get_model_state(llama_self_model_state_info * out_info) const;
    int32_t self_state_trace_count() const;
    bool self_state_clear_trace();
    bool self_state_replay_trace(int32_t upto_count);
    bool self_state_replay_trace_on_channel(int32_t upto_count, int32_t replay_channel);
    bool self_state_set_updater_program(const llama_self_updater_program & program);
    bool self_state_get_updater_program(llama_self_updater_program * out_program) const;
    size_t self_state_trace_export_size() const;
    bool self_state_trace_export(void * dst, size_t size) const;
    bool self_state_trace_import(const void * src, size_t size, bool replace_existing);
    bool self_state_evaluate_counterfactual(const llama_self_updater_program & program, int32_t upto_count, llama_self_counterfactual_result * out_result) const;
    bool self_state_evaluate_counterfactual_on_channel(const llama_self_updater_program & program, int32_t upto_count, int32_t replay_channel, llama_self_counterfactual_result * out_result) const;
    bool self_state_build_prewrite_features(const llama_self_state_event & event, llama_self_state_feature_vector * out_features) const;
    bool self_state_apply_prewrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features);
    bool self_state_build_postwrite_features(const llama_self_state_event & event, llama_self_state_feature_vector * out_features) const;
    bool self_state_apply_postwrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features);
    bool hard_memory_configure(const llama_hard_memory_config & config);
    bool hard_memory_get_config(llama_hard_memory_config * out_config) const;
    bool hard_memory_query(const llama_hard_memory_query_request & query, llama_hard_memory_result * out_result);
    bool hard_memory_get_last_result(llama_hard_memory_result * out_result) const;
    bool hard_memory_get_last_archive_trace(llama_hard_memory_archive_trace * out_trace) const;
    bool bash_tool_configure(const llama_bash_tool_config & config);
    bool bash_tool_get_config(llama_bash_tool_config * out_config) const;
    bool bash_tool_set_request(const llama_bash_tool_request & request);
    bool bash_tool_clear_request(int32_t command_id);
    bool bash_tool_submit_result(const llama_bash_tool_result & result);
    bool bash_tool_get_last_result(llama_bash_tool_result * out_result) const;
    bool cognitive_bash_tool_get_request(int32_t command_id, llama_bash_tool_request * out_request) const;
    bool cognitive_bash_tool_submit_result(const llama_bash_tool_result & result, llama_active_loop_trace * out_active_trace);
    int32_t cognitive_tool_spec_count() const;
    bool cognitive_tool_spec_get(int32_t index, llama_cognitive_tool_spec * out_spec) const;
    bool cognitive_tool_spec_set(const llama_cognitive_tool_spec * specs, int32_t count);
    int32_t cognitive_command_count() const;
    bool cognitive_command_get(int32_t index, llama_cognitive_command * out_command) const;
    bool cognitive_command_ack(int32_t command_id);
    bool cognitive_command_complete(int32_t command_id, bool cancelled);
    bool cognitive_active_runner_get(llama_cognitive_active_runner_status * out_status) const;
    bool cognitive_dmn_runner_get(llama_cognitive_dmn_runner_status * out_status) const;
    bool active_loop_process(const llama_self_state_event & event, llama_active_loop_trace * out_trace);
    bool active_loop_note_emit(int32_t episode_id, size_t emitted_text_bytes);
    bool active_loop_get_last_trace(llama_active_loop_trace * out_trace) const;
    bool dmn_tick(uint64_t now_us, llama_dmn_tick_trace * out_trace);
    bool dmn_defer(uint64_t now_us, llama_dmn_tick_trace * out_trace);
    bool dmn_get_last_trace(llama_dmn_tick_trace * out_trace) const;
    bool cognitive_host_state(llama_cognitive_host_state * out_state) const;
    bool favorable_state_get(llama_favorable_state_profile * out_profile) const;
    bool counterfactual_get_last_trace(llama_counterfactual_trace * out_trace) const;
    bool remediation_get_last_plan(llama_remediation_plan * out_plan) const;
    bool governance_get_last_trace(llama_governance_trace * out_trace) const;

    void attach_adapter_runtime(llama_adapter_lora * adapter, float scale, llama_adapter_lora_layer_role role);
    void detach_adapter_runtime(llama_adapter_lora * adapter);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    friend class llama_active_lora_manager;

    void rebuild_lora_stack();
    void log_lora_stack() const;

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr        cvec;
    llama_adapter_lora_stack_ptr  request_loras;
    llama_adapter_lora_stack_ptr  runtime_loras;
    llama_adapter_lora_stack_ptr  loras;
    uint64_t                      lora_stack_version = 0;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    std::unique_ptr<llama_active_lora_manager> active_lora_manager;
    std::unique_ptr<llama_self_state> self_state;
    std::unique_ptr<llama_cognitive_loop> cognitive_loop;
    std::unique_ptr<llama_hard_memory> hard_memory;
    std::unique_ptr<llama_bash_tool> bash_tool;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
