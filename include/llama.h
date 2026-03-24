#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_TOKEN_NULL -1

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 9

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#    define LLAMA_CPP_MEMBER_INIT(value) = value
#else
#    define LLAMA_CPP_MEMBER_INIT(value)
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;

    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLAMA_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE   = -1,
        LLAMA_ROPE_TYPE_NORM   = 0,
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    enum llama_token_attr {
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LLAMA_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llama_flash_attn_type {
        LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    LLAMA_API const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type);

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE  = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW   = 2, // split layers and KV across GPUs, use tensor parallelism if supported
    };

    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void * user_data);

    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
        LLAMA_KV_OVERRIDE_TYPE_STR,
    };

    enum llama_model_meta_key {
        LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_K,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_P,
        LLAMA_MODEL_META_KEY_SAMPLING_MIN_P,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LLAMA_MODEL_META_KEY_SAMPLING_TEMP,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    struct llama_model_kv_override {
        enum llama_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llama_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct llama_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs

        // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llama_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool use_mmap;        // use mmap if possible
        bool use_direct_io;   // use direct io, takes precedence over use_mmap when supported
        bool use_mlock;       // force system to keep model in RAM
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
    };

    struct llama_sampler_seq_config {
        llama_seq_id           seq_id;
        struct llama_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/llama.cpp/pull/7544
    struct llama_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
        enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum llama_attention_type    attention_type;    // attention type to use for embeddings
        enum llama_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        // ref: https://github.com/ggml-org/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of llama_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: https://github.com/ggml-org/llama.cpp/pull/14363

        // [EXPERIMENTAL]
        // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
        // note: the samplers must be sampler chains (i.e. use llama_sampler_chain_init)
        struct llama_sampler_seq_config * samplers;
        size_t                            n_samplers;
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int32_t nthread;                      // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;               // quantize to this llama_ftype
        enum ggml_type output_tensor_type;    // output tensor type
        enum ggml_type token_embedding_type;  // token embeddings tensor type
        bool allow_requantize;                // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;          // quantize output.weight
        bool only_copy;                       // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                            // quantize all tensors to the default type
        bool keep_split;                      // quantize to the same number of shards
        bool dry_run;                         // calculate and show the final quantization size without performing quantization
        void * imatrix;                       // pointer to importance matrix data
        void * kv_overrides;                  // pointer to vector containing overrides
        void * tensor_types;                  // pointer to vector containing tensor types
        void * prune_layers;                  // pointer to vector containing layer indices to prune
    } llama_model_quantize_params;

    typedef struct llama_logit_bias {
        llama_token token;
        float bias;
    } llama_logit_bias;

    typedef struct llama_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } llama_sampler_chain_params;

    // used in chat template
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    // lora adapter
    struct llama_adapter_lora;

    enum llama_active_lora_embedding_type {
        LLAMA_ACTIVE_LORA_EMBEDDING_HASH       = 0,
        LLAMA_ACTIVE_LORA_EMBEDDING_TOKEN_POOL = 1,
        LLAMA_ACTIVE_LORA_EMBEDDING_HIDDEN_STATE = 2,
    };

    enum llama_memory_lora_bucket {
        LLAMA_MEMORY_LORA_BUCKET_PAST_WEEK    = 0,
        LLAMA_MEMORY_LORA_BUCKET_PAST_MONTH   = 1,
        LLAMA_MEMORY_LORA_BUCKET_PAST_QUARTER = 2,
        LLAMA_MEMORY_LORA_BUCKET_PAST_YEAR    = 3,
        LLAMA_MEMORY_LORA_BUCKET_ALL_TIME     = 4,
        LLAMA_MEMORY_LORA_BUCKET_COUNT        = 5,
    };

    enum llama_serving_lora_layer_role {
        LLAMA_SERVING_LORA_LAYER_REQUEST      = 0,
        LLAMA_SERVING_LORA_LAYER_ALL_TIME     = 1,
        LLAMA_SERVING_LORA_LAYER_PAST_YEAR    = 2,
        LLAMA_SERVING_LORA_LAYER_PAST_QUARTER = 3,
        LLAMA_SERVING_LORA_LAYER_PAST_MONTH   = 4,
        LLAMA_SERVING_LORA_LAYER_PAST_WEEK    = 5,
        LLAMA_SERVING_LORA_LAYER_ACTIVE       = 6,
        LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_TOOL_SELECTION  = 7,
        LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_PLANNING_COMPOSITION = 8,
        LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_COUNTERFACTUAL  = 9,
        LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_MEMORY_COMPRESS = 10,
        LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_SELF_OBSERVE    = 11,
        LLAMA_SERVING_LORA_LAYER_USER_PERSONALITY           = 12,
        LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_PROCESS_LEARNED = 13,
        LLAMA_SERVING_LORA_LAYER_FUNCTIONAL_PROCESS_BOOTSTRAP = 14,
    };

    enum llama_functional_lora_family {
        LLAMA_FUNCTIONAL_LORA_TOOL_SELECTION = 0,
        LLAMA_FUNCTIONAL_LORA_PLANNING_COMPOSITION = 1,
        LLAMA_FUNCTIONAL_LORA_COUNTERFACTUAL = 2,
        LLAMA_FUNCTIONAL_LORA_MEMORY_COMPRESSION = 3,
        LLAMA_FUNCTIONAL_LORA_SELF_OBSERVATION = 4,
        LLAMA_FUNCTIONAL_LORA_COUNT = 5,
    };

    enum llama_functional_microphase {
        LLAMA_FUNCTIONAL_MICROPHASE_NONE = 0,
        LLAMA_FUNCTIONAL_MICROPHASE_STATE_INTERPRET = 1,
        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_CLASS_SELECTION = 2,
        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_ARGUMENT_PREP = 3,
        LLAMA_FUNCTIONAL_MICROPHASE_TOOL_RESULT_INTEGRATION = 4,
        LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_GENERATE = 5,
        LLAMA_FUNCTIONAL_MICROPHASE_COUNTERFACTUAL_COMPARE = 6,
        LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_COMPRESSION = 7,
        LLAMA_FUNCTIONAL_MICROPHASE_MEMORY_AUDIT = 8,
        LLAMA_FUNCTIONAL_MICROPHASE_SELF_OBSERVE = 9,
        LLAMA_FUNCTIONAL_MICROPHASE_SELF_FORECAST = 10,
        LLAMA_FUNCTIONAL_MICROPHASE_POST_ACTION_REFLECTION = 11,
        LLAMA_FUNCTIONAL_MICROPHASE_PLAN_DRAFT = 12,
        LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE = 13,
        LLAMA_FUNCTIONAL_MICROPHASE_PLAN_REVISE = 14,
    };

    enum llama_functional_hold_unit {
        LLAMA_FUNCTIONAL_HOLD_PHASE_EXIT = 0,
        LLAMA_FUNCTIONAL_HOLD_LOOP_STEPS = 1,
        LLAMA_FUNCTIONAL_HOLD_COMMANDS = 2,
        LLAMA_FUNCTIONAL_HOLD_TOKENS = 3,
    };

    enum llama_functional_snapshot_source {
        LLAMA_FUNCTIONAL_SNAPSHOT_SOURCE_WEEKLY_ARCHIVE = 0,
        LLAMA_FUNCTIONAL_SNAPSHOT_SOURCE_PERSIST_RESTORE = 1,
        LLAMA_FUNCTIONAL_SNAPSHOT_SOURCE_MANUAL_RESEED = 2,
    };

    enum llama_functional_bias_proposal_family {
        LLAMA_FUNCTIONAL_BIAS_PROPOSAL_LOCAL = 0,
        LLAMA_FUNCTIONAL_BIAS_PROPOSAL_HISTORICAL = 1,
        LLAMA_FUNCTIONAL_BIAS_PROPOSAL_ORTHOGONAL = 2,
    };

    enum llama_functional_lora_target_kind {
        LLAMA_FUNCTIONAL_LORA_TARGET_FAMILY_ROOT = 0,
        LLAMA_FUNCTIONAL_LORA_TARGET_PROCESS_ENTRY = 1,
    };

    enum llama_functional_replay_mode {
        LLAMA_FUNCTIONAL_REPLAY_MODE_NONE = 0,
        LLAMA_FUNCTIONAL_REPLAY_MODE_CURRENT = 1,
        LLAMA_FUNCTIONAL_REPLAY_MODE_ARCHIVED = 2,
        LLAMA_FUNCTIONAL_REPLAY_MODE_LOCAL_PERTURBED = 3,
        LLAMA_FUNCTIONAL_REPLAY_MODE_ORTHOGONAL = 4,
    };

    enum llama_functional_route_reason_flags {
        LLAMA_FUNCTIONAL_ROUTE_REASON_UNCERTAINTY = 1u << 0,
        LLAMA_FUNCTIONAL_ROUTE_REASON_TOOL_AFFINITY = 1u << 1,
        LLAMA_FUNCTIONAL_ROUTE_REASON_FAVORABLE_DIVERGENCE = 1u << 2,
        LLAMA_FUNCTIONAL_ROUTE_REASON_MEMORY_PRESSURE = 1u << 3,
        LLAMA_FUNCTIONAL_ROUTE_REASON_PREDICTION_ERROR = 1u << 4,
        LLAMA_FUNCTIONAL_ROUTE_REASON_HOLD_CONTINUITY = 1u << 5,
    };

    typedef bool (*llama_active_lora_embedding_callback)(
            const struct llama_context * ctx,
            const llama_token * tokens,
            size_t n_tokens,
            float * out_embedding,
            size_t n_embedding,
            void * user_data);

    struct llama_active_lora_params {
        bool     enabled;
        float    host_memory_ratio;
        float    device_memory_ratio;
        uint32_t min_rank;
        uint32_t max_rank;
        uint32_t train_context_tokens;
        uint32_t train_stride_tokens;
        uint32_t max_updates_before_rollover;
        float    adapter_scale;
        float    learning_rate;
        float    weight_decay;
        float    gain_max;
        float    gain_decay;
        uint32_t embedding_dim;
        int32_t  embedding_type;
        llama_active_lora_embedding_callback embedding_callback;
        void *   embedding_callback_user_data;
    };

    struct llama_active_lora_stats {
        bool     enabled;
        bool     rollover_ready;
        uint32_t selected_rank;
        uint32_t updates_applied;
        uint64_t optimizer_step_count;
        uint64_t tokens_ingested;
        float    host_memory_ratio;
        float    device_memory_ratio;
        uint64_t host_budget_bytes;
        uint64_t device_budget_bytes;
        uint32_t embedding_dim;
        bool     embedding_is_custom;
        float    gain_mean;
        float    gain_max;
        float    optimizer_last_update_norm;
        int32_t  embedding_type;
    };

    struct llama_user_personality_lora_stats {
        bool     enabled;
        bool     attached_for_simulation;
        uint32_t selected_rank;
        uint32_t updates_applied;
        uint64_t optimizer_step_count;
        uint64_t tokens_ingested;
        float    adapter_scale;
        float    gain_mean;
        float    gain_max;
        float    optimizer_last_update_norm;
        float    confidence;
    };

    struct llama_past_lora_params {
        bool     enabled;
        float    host_memory_ratio[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        float    device_memory_ratio[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        uint32_t min_rank[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        uint32_t max_rank[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        float    base_scale[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        uint64_t decay_half_life_us[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        uint64_t condensation_period_us[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        float    merge_source_weight[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        float    merge_target_retention[LLAMA_MEMORY_LORA_BUCKET_COUNT];
        float    gain_max;
        float    gain_decay;
        float    singular_value_floor;
    };

    struct llama_past_lora_bucket_stats {
        bool     populated;
        uint32_t version;
        uint32_t selected_rank;
        uint64_t created_at_us;
        uint64_t source_window_start_us;
        uint64_t source_window_end_us;
        uint64_t host_budget_bytes;
        uint64_t device_budget_bytes;
        float    base_scale;
        float    effective_scale;
        float    gain_mean;
        float    gain_max;
    };

    struct llama_past_lora_stats {
        bool     enabled;
        uint64_t last_tick_us;
        uint64_t pending_job_mask;
        struct llama_past_lora_bucket_stats buckets[LLAMA_MEMORY_LORA_BUCKET_COUNT];
    };

    struct llama_serving_lora_layer_info {
        float   scale;
        int32_t precedence;
        int32_t role;
    };

    struct llama_functional_lora_family_config {
        bool enabled;
        uint32_t rank_min;
        uint32_t rank_max;
        float gain_min;
        float gain_max;
        float gain_clip_min;
        float gain_clip_max;
        float default_gain;
        float exploration_noise_initial_std;
        float exploration_noise_min_std;
        uint32_t exploration_noise_decay_invocations;
        float bootstrap_perturbation_initial_std;
        float bootstrap_perturbation_min_std;
        uint32_t bootstrap_perturbation_decay_activations;
        float bootstrap_weight_init_std;
        float negative_update_scale;
        float positive_update_scale;
        uint32_t top_k_priority;
        uint32_t update_horizon_steps;
        uint32_t update_horizon_commands;
        bool allow_active_loop;
        bool allow_dmn_loop;
    };

    struct llama_functional_outcome_snapshot {
        float favorable_divergence;
        float user_satisfaction_risk;
        float goal_progress_pressure;
        float loop_inefficiency;
        float recovery_urgency;
        float answerability;
        float preference_uncertainty;
        float expected_steps_remaining;
        float expected_inference_cost_remaining;
    };

    struct llama_functional_lora_update_info {
        bool valid;
        int32_t family;
        int32_t loop_origin;
        int32_t start_microphase;
        int32_t settle_microphase;
        int32_t selected_tool_kind;
        int32_t candidate_count;
        float signed_outcome;
        float magnitude;
        uint64_t source_token_hash;
        int32_t source_token_count;
        float metrics[6];
        float meta_loss;
        float allostatic_distance_before;
        float allostatic_distance_after;
        float exploration_std;
        float parameter_update_norm;
        uint64_t adam_step;
        float adapter_update_norm;
        uint64_t adapter_optimizer_step;
        struct llama_functional_outcome_snapshot before_snapshot;
        struct llama_functional_outcome_snapshot after_snapshot;
    };

    struct llama_functional_hold_state {
        bool active;
        int32_t family;
        int32_t loop_origin;
        int32_t microphase_started;
        int32_t microphase_current;
        int32_t hold_unit;
        uint32_t hold_budget;
        uint32_t hold_remaining;
        float gain;
        float bootstrap_std;
        float bootstrap_perturbation;
        bool replay_boundary_required;
    };

    struct llama_functional_activation_decision {
        int32_t loop_origin;
        int32_t microphase;
        uint64_t activated_mask;
        uint64_t eligible_mask;
        int32_t top_family;
        int32_t family_count;
        float gains[LLAMA_FUNCTIONAL_LORA_COUNT];
        float predicted_gains[LLAMA_FUNCTIONAL_LORA_COUNT];
        float sampled_noise[LLAMA_FUNCTIONAL_LORA_COUNT];
        float bootstrap_std[LLAMA_FUNCTIONAL_LORA_COUNT];
        float bootstrap_perturbation[LLAMA_FUNCTIONAL_LORA_COUNT];
        int32_t hold_unit[LLAMA_FUNCTIONAL_LORA_COUNT];
        uint32_t hold_value[LLAMA_FUNCTIONAL_LORA_COUNT];
        float priority[LLAMA_FUNCTIONAL_LORA_COUNT];
        uint32_t reason_mask[LLAMA_FUNCTIONAL_LORA_COUNT];
        float exploration_std;
        float allostatic_distance;
        float allostatic_gradient_norm;
        uint64_t gating_invocation_count;
    };

    struct llama_functional_lora_family_state {
        int32_t family;
        bool enabled;
        bool compatible;
        bool active_now;
        float current_gain;
        float predicted_gain;
        float last_noise;
        float current_bootstrap_std;
        float last_bootstrap_perturbation;
        int32_t current_microphase;
        int32_t current_hold_unit;
        uint32_t current_hold_remaining;
        uint64_t activation_count;
        uint64_t update_count;
        float last_signed_outcome;
        float last_meta_loss;
    };

    struct llama_functional_lora_trace {
        struct llama_functional_activation_decision last_activation;
        struct llama_functional_hold_state holds[LLAMA_FUNCTIONAL_LORA_COUNT];
        struct llama_functional_lora_family_state family_state[LLAMA_FUNCTIONAL_LORA_COUNT];
    };

    struct llama_functional_lora_snapshot_info {
        bool valid;
        int32_t family;
        int32_t slot;
        int32_t source;
        uint64_t snapshot_id;
        uint64_t captured_at_us;
        uint64_t expires_at_us;
        uint64_t source_update_count;
        float self_state_gradient_norm;
        float robustness_score;
        float last_signed_outcome;
        float dominant_direction_cosine;
    };

    struct llama_functional_lora_snapshot_archive {
        int32_t family;
        uint32_t count;
        uint64_t last_capture_us;
        uint64_t next_capture_due_us;
        struct llama_functional_lora_snapshot_info items[4];
    };

    struct llama_functional_snapshot_maintenance_trace {
        bool ran;
        bool captured_any;
        uint64_t now_us;
        uint64_t next_due_us;
        uint32_t expired_count;
        uint32_t captured_count;
        uint64_t captured_family_mask;
    };

    struct llama_functional_lora_replay_override {
        bool active;
        int32_t family;
        int32_t replay_mode;
        int32_t snapshot_slot;
        float replay_gain;
        float perturbation_scale;
        float cosine_limit;
        bool disable_bootstrap;
    };

    struct llama_functional_lora_differential_update {
        bool valid;
        int32_t family;
        int32_t proposal_family;
        int32_t source_snapshot_slot;
        float signed_score_delta;
        float magnitude;
        float lora_difference_norm;
        float applied_update_norm;
        float robustness_score;
        uint64_t adapter_optimizer_step;
    };

    struct llama_functional_lora_ablation_config {
        uint64_t disabled_family_mask;
        uint64_t disabled_microphase_mask;
        bool disable_functional_stack;
        bool disable_update_writes;
        bool disable_hold_windows;
    };

    enum llama_temporal_self_improvement_outcome {
        LLAMA_TEMPORAL_SELF_IMPROVEMENT_NONE    = 0,
        LLAMA_TEMPORAL_SELF_IMPROVEMENT_REWARD  = 1,
        LLAMA_TEMPORAL_SELF_IMPROVEMENT_DAMPEN  = 2,
        LLAMA_TEMPORAL_SELF_IMPROVEMENT_TIE     = 3,
        LLAMA_TEMPORAL_SELF_IMPROVEMENT_SKIPPED = 4,
    };

    struct llama_active_temporal_encoding_bias {
        float reward_bias;
        float dampening_bias;
        float effective_write_scale;
        float last_update_norm;
        uint64_t adam_step;
        uint64_t applied_update_count;
        int64_t last_update_monotonic_ms;
    };

    struct llama_temporal_self_improvement_trace {
        bool valid;
        int32_t loop_origin;
        int32_t selected_temporal_role;
        int32_t counterfactual_family;
        int32_t outcome;
        float evolution_uncertainty_before;
        float evolution_uncertainty_after;
        float signed_advantage;
        float efficiency_advantage;
        float active_reward_bias;
        float active_dampening_bias;
        float active_effective_write_scale;
    };

    enum llama_self_register_family {
        LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR = 0,
        LLAMA_SELF_REGISTER_FAMILY_SIGNED_SCALAR  = 1,
        LLAMA_SELF_REGISTER_FAMILY_CATEGORICAL    = 2,
    };

    enum llama_self_register_id {
        LLAMA_SELF_REGISTER_UNCERTAINTY = 0,
        LLAMA_SELF_REGISTER_CONTRADICTION,
        LLAMA_SELF_REGISTER_NOVELTY,
        LLAMA_SELF_REGISTER_TOPIC_SHIFT,
        LLAMA_SELF_REGISTER_GOAL_RELEVANCE,
        LLAMA_SELF_REGISTER_SELF_RELEVANCE,
        LLAMA_SELF_REGISTER_SOCIAL_RELEVANCE,
        LLAMA_SELF_REGISTER_AFFORDANCE,
        LLAMA_SELF_REGISTER_BROADCAST_PRESSURE,
        LLAMA_SELF_REGISTER_BROADCAST_INHIBITION,
        LLAMA_SELF_REGISTER_FOLLOWUP_CONTINUATION,
        LLAMA_SELF_REGISTER_MEMORY_WRITE_PRIORITY,
        LLAMA_SELF_REGISTER_TIME_PHASE,
        LLAMA_SELF_REGISTER_TOOL_SALIENCE,
        LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK,
        LLAMA_SELF_REGISTER_GOAL_PROGRESS_PRESSURE,
        LLAMA_SELF_REGISTER_LOOP_INEFFICIENCY,
        LLAMA_SELF_REGISTER_RECOVERY_URGENCY,
        LLAMA_SELF_REGISTER_ANSWERABILITY,
        LLAMA_SELF_REGISTER_PREFERENCE_UNCERTAINTY,
        LLAMA_SELF_REGISTER_USER_DIRECTNESS_PREFERENCE,
        LLAMA_SELF_REGISTER_USER_VERBOSITY_PREFERENCE,
        LLAMA_SELF_REGISTER_USER_STRUCTURE_PREFERENCE,
        LLAMA_SELF_REGISTER_USER_AUTONOMY_PREFERENCE,
        LLAMA_SELF_REGISTER_USER_CLARIFICATION_PREFERENCE,
        LLAMA_SELF_REGISTER_USER_DISAGREEMENT_SENSITIVITY,
        LLAMA_SELF_REGISTER_EVOLUTION_UNCERTAINTY,
        LLAMA_SELF_REGISTER_DISCOVERED_STATE_LOAD,
        LLAMA_SELF_REGISTER_DISCOVERED_STATE_PERMANENCE,
        LLAMA_SELF_REGISTER_DISCOVERED_STATE_ALLOSTATIC_LOAD,
        LLAMA_SELF_REGISTER_CHANNEL_STATE,
        LLAMA_SELF_REGISTER_COUNT,
    };

    enum llama_self_state_channel_state {
        LLAMA_SELF_STATE_CHANNEL_WAITING          = 0,
        LLAMA_SELF_STATE_CHANNEL_ACTIVE           = 1,
        LLAMA_SELF_STATE_CHANNEL_DO_NOT_INTERRUPT = 2,
    };

    enum llama_self_state_event_role {
        LLAMA_SELF_STATE_EVENT_USER   = 0,
        LLAMA_SELF_STATE_EVENT_TOOL   = 1,
        LLAMA_SELF_STATE_EVENT_SYSTEM = 2,
    };

    enum llama_self_state_event_channel {
        LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY        = 0,
        LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL = 1,
    };

    enum llama_self_state_source_flags {
        LLAMA_SELF_SOURCE_INIT          = 1u << 0,
        LLAMA_SELF_SOURCE_TIME          = 1u << 1,
        LLAMA_SELF_SOURCE_USER_EVENT    = 1u << 2,
        LLAMA_SELF_SOURCE_TOOL_EVENT    = 1u << 3,
        LLAMA_SELF_SOURCE_EMIT_EVENT    = 1u << 4,
        LLAMA_SELF_SOURCE_CHANNEL       = 1u << 5,
        LLAMA_SELF_SOURCE_EXTERNAL_TIME = 1u << 6,
        LLAMA_SELF_SOURCE_COUNTERFACTUAL = 1u << 7,
        LLAMA_SELF_SOURCE_INTERNAL_ARTIFACT = 1u << 8,
    };

    enum llama_self_state_event_flags {
        LLAMA_SELF_STATE_EVENT_ADMITTED       = 1u << 0,
        LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED = 1u << 1,
        LLAMA_SELF_STATE_EVENT_TOOL_FAILED    = 1u << 2,
        LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP  = 1u << 3,
        LLAMA_SELF_STATE_EVENT_INTERNAL_ARTIFACT = 1u << 4,
        LLAMA_SELF_STATE_EVENT_CONTEXT_COMPACTED = 1u << 5,
    };

    enum llama_self_cognitive_artifact_kind {
        LLAMA_SELF_COG_ARTIFACT_EXTERNAL_EVENT      = 0,
        LLAMA_SELF_COG_ARTIFACT_ACTIVE_PLAN         = 1,
        LLAMA_SELF_COG_ARTIFACT_ACTIVE_REFLECTION   = 2,
        LLAMA_SELF_COG_ARTIFACT_DMN_PLAN            = 3,
        LLAMA_SELF_COG_ARTIFACT_DMN_INTERNAL_WRITE  = 4,
        LLAMA_SELF_COG_ARTIFACT_GOVERNANCE          = 5,
        LLAMA_SELF_COG_ARTIFACT_FUNCTIONAL_UPDATE   = 6,
        LLAMA_SELF_COG_ARTIFACT_CONTEXT_EVICTION    = 7,
        LLAMA_SELF_COG_ARTIFACT_HIDDEN_THOUGHT      = 8,
        LLAMA_SELF_COG_ARTIFACT_TOOL_CALL           = 9,
        LLAMA_SELF_COG_ARTIFACT_TOOL_OBSERVATION    = 10,
        LLAMA_SELF_COG_ARTIFACT_VISIBLE_OUTPUT      = 11,
        LLAMA_SELF_COG_ARTIFACT_EMOTIVE_MOMENT      = 12,
        LLAMA_SELF_COG_ARTIFACT_INTERNAL_SUMMARY    = 13,
    };

    enum llama_shared_cognitive_context_origin {
        LLAMA_SHARED_CONTEXT_ORIGIN_ACTIVE = 0,
        LLAMA_SHARED_CONTEXT_ORIGIN_DMN    = 1,
        LLAMA_SHARED_CONTEXT_ORIGIN_TOOL   = 2,
        LLAMA_SHARED_CONTEXT_ORIGIN_SYSTEM = 3,
    };

    enum llama_shared_cognitive_context_kind {
        LLAMA_SHARED_CONTEXT_KIND_USER_MESSAGE     = 0,
        LLAMA_SHARED_CONTEXT_KIND_HIDDEN_THOUGHT   = 1,
        LLAMA_SHARED_CONTEXT_KIND_TOOL_CALL        = 2,
        LLAMA_SHARED_CONTEXT_KIND_TOOL_OBSERVATION = 3,
        LLAMA_SHARED_CONTEXT_KIND_VISIBLE_OUTPUT   = 4,
        LLAMA_SHARED_CONTEXT_KIND_EMOTIVE_MOMENT   = 5,
        LLAMA_SHARED_CONTEXT_KIND_CONTEXT_EVICTION = 6,
        LLAMA_SHARED_CONTEXT_KIND_INTERNAL_SUMMARY = 7,
    };

    enum llama_shared_cognitive_context_phase {
        LLAMA_SHARED_CONTEXT_PHASE_THINK          = 0,
        LLAMA_SHARED_CONTEXT_PHASE_SELECT_TOOL    = 1,
        LLAMA_SHARED_CONTEXT_PHASE_PREPARE_TOOL   = 2,
        LLAMA_SHARED_CONTEXT_PHASE_OBSERVE        = 3,
        LLAMA_SHARED_CONTEXT_PHASE_EMIT           = 4,
        LLAMA_SHARED_CONTEXT_PHASE_COMPRESS       = 5,
        LLAMA_SHARED_CONTEXT_PHASE_COUNTERFACTUAL = 6,
    };

    enum llama_self_tool_job_status {
        LLAMA_SELF_TOOL_JOB_IDLE      = 0,
        LLAMA_SELF_TOOL_JOB_PENDING   = 1,
        LLAMA_SELF_TOOL_JOB_RUNNING   = 2,
        LLAMA_SELF_TOOL_JOB_COMPLETED = 3,
        LLAMA_SELF_TOOL_JOB_FAILED    = 4,
    };

    enum llama_tool_kind {
        LLAMA_TOOL_KIND_NONE              = 0,
        LLAMA_TOOL_KIND_GENERIC           = 1,
        LLAMA_TOOL_KIND_HARD_MEMORY_QUERY = 2,
        LLAMA_TOOL_KIND_HARD_MEMORY_WRITE = 3,
        LLAMA_TOOL_KIND_BASH_CLI          = 4,
        LLAMA_TOOL_KIND_CODEX_CLI         = 5,
        LLAMA_TOOL_KIND_TELEGRAM_RELAY    = 6,
        LLAMA_TOOL_KIND_TELEGRAM_ASK_OPTIONS = 7,
    };

    enum llama_self_model_extension_kind {
        LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT = 0,
        LLAMA_SELF_MODEL_EXTENSION_SCALAR_PARAM   = 1,
    };

    enum llama_self_model_extension_source {
        LLAMA_SELF_MODEL_EXTENSION_SOURCE_COUNTERFACTUAL = 0,
        LLAMA_SELF_MODEL_EXTENSION_SOURCE_EVENT_FEEDBACK = 1,
        LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_HARD_MEMORY = 2,
        LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_BASH_CLI = 3,
        LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_EXTERNAL = 4,
    };

    enum llama_self_model_extension_domain {
        LLAMA_SELF_MODEL_EXTENSION_DOMAIN_GOAL_PROGRESS    = 0,
        LLAMA_SELF_MODEL_EXTENSION_DOMAIN_USER_OUTCOME     = 1,
        LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EPISTEMIC        = 2,
        LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EFFICIENCY       = 3,
        LLAMA_SELF_MODEL_EXTENSION_DOMAIN_RECOVERY         = 4,
        LLAMA_SELF_MODEL_EXTENSION_DOMAIN_STRATEGY         = 5,
        LLAMA_SELF_MODEL_EXTENSION_DOMAIN_SELF_IMPROVEMENT = 6,
    };

    enum llama_self_model_extension_flags {
        LLAMA_SELF_MODEL_EXTENSION_FLAG_ACTIVE            = 1u << 0,
        LLAMA_SELF_MODEL_EXTENSION_FLAG_HAS_DESIRED_STATE = 1u << 1,
        LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN       = 1u << 2,
        LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS = 1u << 3,
        LLAMA_SELF_MODEL_EXTENSION_FLAG_DISCOVERED        = 1u << 4,
    };

    enum llama_self_model_extension_stage {
        LLAMA_SELF_MODEL_EXTENSION_STAGE_TRANSIENT  = 0,
        LLAMA_SELF_MODEL_EXTENSION_STAGE_PERMANENT  = 1,
        LLAMA_SELF_MODEL_EXTENSION_STAGE_ALLOSTATIC = 2,
    };

    enum llama_self_memory_handle_kind {
        LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER = 0,
        LLAMA_SELF_MEMORY_HANDLE_FROZEN_BUCKET          = 1,
        LLAMA_SELF_MEMORY_HANDLE_ACTIVE_MEMORY          = 2,
        LLAMA_SELF_MEMORY_HANDLE_EXTERNAL               = 3,
    };

    enum llama_self_updater_phase_mask {
        LLAMA_SELF_UPDATER_PHASE_PREWRITE  = 1u << 0,
        LLAMA_SELF_UPDATER_PHASE_POSTWRITE = 1u << 1,
    };

    enum llama_self_updater_feature_id {
        LLAMA_SELF_UPDATER_FEATURE_NONE = -1,
        LLAMA_SELF_UPDATER_FEATURE_NOVELTY = 0,
        LLAMA_SELF_UPDATER_FEATURE_TOPIC_SHIFT = 1,
        LLAMA_SELF_UPDATER_FEATURE_GOAL_SIMILARITY = 2,
        LLAMA_SELF_UPDATER_FEATURE_COMMITMENT_SIMILARITY = 3,
        LLAMA_SELF_UPDATER_FEATURE_IDENTITY_SIMILARITY = 4,
        LLAMA_SELF_UPDATER_FEATURE_SELF_REFERENCE = 5,
        LLAMA_SELF_UPDATER_FEATURE_NEGATION_RATIO = 6,
        LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY_LEXICAL = 7,
        LLAMA_SELF_UPDATER_FEATURE_ERROR_RATIO = 8,
        LLAMA_SELF_UPDATER_FEATURE_RECENCY_USER = 9,
        LLAMA_SELF_UPDATER_FEATURE_RECENCY_TOOL = 10,
        LLAMA_SELF_UPDATER_FEATURE_RECENCY_EMIT = 11,
        LLAMA_SELF_UPDATER_FEATURE_SOCIAL_FAMILIARITY = 12,
        LLAMA_SELF_UPDATER_FEATURE_SOCIAL_TRUST = 13,
        LLAMA_SELF_UPDATER_FEATURE_SOCIAL_RECIPROCITY = 14,
        LLAMA_SELF_UPDATER_FEATURE_SOCIAL_BOND = 15,
        LLAMA_SELF_UPDATER_FEATURE_TOOL_READINESS = 16,
        LLAMA_SELF_UPDATER_FEATURE_TOOL_PENDING_PRESSURE = 17,
        LLAMA_SELF_UPDATER_FEATURE_DECODER_ENTROPY = 18,
        LLAMA_SELF_UPDATER_FEATURE_DECODER_TOP_MARGIN = 19,
        LLAMA_SELF_UPDATER_FEATURE_CONTRADICTION = 20,
        LLAMA_SELF_UPDATER_FEATURE_UNCERTAINTY = 21,
        LLAMA_SELF_UPDATER_FEATURE_MEMORY_WRITE_PRESSURE = 22,
        LLAMA_SELF_UPDATER_FEATURE_BROADCAST_PRESSURE_HINT = 23,
        LLAMA_SELF_UPDATER_FEATURE_BROADCAST_INHIBITION_HINT = 24,
        LLAMA_SELF_UPDATER_FEATURE_FOLLOWUP_HINT = 25,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_USER = 26,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_TOOL = 27,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_ROLE_SYSTEM = 28,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_CHANNEL_PRIMARY = 29,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_CHANNEL_COUNTERFACTUAL = 30,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_ADMITTED = 31,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_FAILED = 32,
        LLAMA_SELF_UPDATER_FEATURE_EVENT_TOOL_COMPLETED = 33,
    };

    enum {
        LLAMA_SELF_MAX_UPDATER_RULE_TERMS = 8,
        LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS = 4,
        LLAMA_SELF_MAX_UPDATER_RULES = 16,
    };

    struct llama_self_state_time_point {
        int64_t wall_clock_ms;
        int64_t monotonic_ms;
        int32_t timezone_offset_minutes;
    };

    struct llama_self_state_datetime {
        int64_t wall_clock_ms;
        int64_t monotonic_ms;
        int32_t timezone_offset_minutes;
        int32_t local_year;
        int32_t local_month;
        int32_t local_day;
        int32_t local_hour;
        int32_t local_minute;
        int32_t local_second;
        int32_t day_of_week;
        int32_t day_of_year;
        float   hour_sin;
        float   hour_cos;
        float   weekday_sin;
        float   weekday_cos;
        float   year_day_sin;
        float   year_day_cos;
        int64_t delta_since_last_user_ms;
        int64_t delta_since_last_tool_event_ms;
        int64_t delta_since_last_emit_ms;
        int64_t session_age_ms;
    };

    struct llama_self_register_info {
        int32_t  register_id;
        int32_t  family;
        float    scalar_value;
        int32_t  categorical_value;
        float    value_min;
        float    value_max;
        float    confidence;
        int64_t  last_update_wall_ms;
        int64_t  last_update_monotonic_ms;
        uint32_t source_mask;
        uint32_t updater_version;
        bool     dirty;
    };

    struct llama_self_state_event {
        const llama_token * tokens;
        size_t   n_tokens;
        int32_t  role;
        int32_t  channel;
        uint32_t flags;
        float    decoder_entropy;
        float    decoder_top_margin;
        int32_t  artifact_kind LLAMA_CPP_MEMBER_INIT(LLAMA_SELF_COG_ARTIFACT_EXTERNAL_EVENT);
        int32_t  loop_origin LLAMA_CPP_MEMBER_INIT(-1);
        int32_t  phase LLAMA_CPP_MEMBER_INIT(-1);
        int32_t  source_id LLAMA_CPP_MEMBER_INIT(-1);
        int32_t  plan_id LLAMA_CPP_MEMBER_INIT(-1);
    };

    struct llama_self_trace_item_info {
        struct llama_self_state_time_point time_point;
        int64_t  context_item_id;
        int32_t  role;
        int32_t  channel;
        uint32_t flags;
        float    decoder_entropy;
        float    decoder_top_margin;
        int32_t  artifact_kind;
        int32_t  context_kind;
        int32_t  loop_origin;
        int32_t  phase;
        int32_t  source_id;
        int32_t  plan_id;
        int32_t  revision_id;
        int32_t  token_count;
    };

    struct llama_shared_cognitive_context_item {
        int64_t  context_item_id;
        int32_t  origin;
        int32_t  kind;
        int32_t  phase;
        int32_t  turn_id;
        int32_t  episode_or_tick_id;
        int32_t  plan_id;
        int32_t  tool_request_id;
        int32_t  token_count;
        uint64_t admitted_at_us;
        int32_t  revision_id;
        uint32_t flags;
        int32_t  role;
        int32_t  channel;
    };

    struct llama_shared_cognitive_context_window {
        int64_t head_revision;
        int32_t item_count;
        int32_t token_count;
        int32_t max_item_count;
        int32_t max_token_count;
        int64_t oldest_item_id;
        int64_t newest_item_id;
        int64_t last_eviction_revision;
        int32_t eviction_count;
    };

    struct llama_self_state_feature_vector {
        float token_count_log;
        float unique_token_ratio;
        float novelty;
        float topic_shift;
        float working_memory_top_similarity;
        float working_memory_similarity_variance;
        float memory_handle_top_similarity;
        float memory_handle_similarity_variance;
        float goal_top_similarity;
        float commitment_top_similarity;
        float identity_similarity;
        float self_reference_ratio;
        float negation_ratio;
        float uncertainty_lexical_ratio;
        float error_ratio;
        float recency_user;
        float recency_tool;
        float recency_emit;
        float social_familiarity;
        float social_trust;
        float social_reciprocity;
        float social_bond_strength;
        float tool_readiness_score;
        float tool_pending_pressure;
        float decoder_entropy;
        float decoder_top_margin;
        float contradiction_score;
        float uncertainty_score;
        float memory_write_pressure;
        float broadcast_pressure_hint;
        float broadcast_inhibition_hint;
        float followup_hint;
        float negative_user_valence;
        float question_ratio;
        float imperative_ratio;
        float list_ratio;
        float newline_ratio;
        float emphasis_ratio;
    };

    typedef bool (*llama_self_state_head_callback)(
            const struct llama_self_state_feature_vector * features,
            float * out_score,
            void * user_data);

    struct llama_self_state_params {
        bool enable_learned_contradiction_head;
        bool enable_learned_uncertainty_head;
        bool enable_learned_broadcast_head;
        bool enable_builtin_contradiction_probe;
        bool enable_builtin_uncertainty_probe;
        bool enable_builtin_broadcast_probe;
        float tool_salience_half_life_ms;
        float prewrite_gain;
        float postwrite_gain;
        bool enable_belief_state;
        int32_t belief_slot_count;
        float belief_residual_decay;
        float belief_pressure_clip;
        float belief_confidence_floor;
        float belief_promotion_threshold;
        float belief_max_update_step;
        float belief_missing_observation_weight;
        float belief_unmodeled_care_weight;
        float belief_forecast_error_weight;
        float belief_counterfactual_miss_weight;
        float belief_memory_residue_weight;
        llama_self_state_head_callback contradiction_head_callback;
        void * contradiction_head_user_data;
        llama_self_state_head_callback uncertainty_head_callback;
        void * uncertainty_head_user_data;
        llama_self_state_head_callback broadcast_head_callback;
        void * broadcast_head_user_data;
    };

    struct llama_self_reactivation_info {
        int32_t handle_id;
        int32_t kind;
        float priority;
        float top_similarity;
        int64_t last_update_monotonic_ms;
    };

    struct llama_self_tool_state_info {
        int32_t active_status;
        int32_t pending_jobs;
        int32_t running_jobs;
        int32_t completed_jobs;
        int32_t failed_jobs;
        float readiness;
        int64_t last_update_monotonic_ms;
    };

    struct llama_self_social_state_info {
        float familiarity;
        float trust;
        float reciprocity;
        float bond_strength;
        float recent_user_valence;
        float dissatisfaction;
        float contact_set_point_hours;
        float silence_hours;
        float silence_deficit;
        int32_t user_turn_count;
        int32_t system_turn_count;
        int64_t last_update_monotonic_ms;
        int64_t last_substantive_contact_monotonic_ms;
    };

    enum llama_self_disturbance_source_kind {
        LLAMA_SELF_DISTURBANCE_SOURCE_NONE = 0,
        LLAMA_SELF_DISTURBANCE_SOURCE_USER_MESSAGE = 1,
        LLAMA_SELF_DISTURBANCE_SOURCE_HIDDEN_THOUGHT = 2,
        LLAMA_SELF_DISTURBANCE_SOURCE_TOOL_SELECTION = 3,
        LLAMA_SELF_DISTURBANCE_SOURCE_TOOL_ARGUMENTS = 4,
        LLAMA_SELF_DISTURBANCE_SOURCE_TOOL_OBSERVATION = 5,
        LLAMA_SELF_DISTURBANCE_SOURCE_VISIBLE_OUTPUT = 6,
        LLAMA_SELF_DISTURBANCE_SOURCE_SOCIAL_SILENCE = 7,
        LLAMA_SELF_DISTURBANCE_SOURCE_DMN_EMIT = 8,
    };

    enum llama_self_disturbance_failure_class {
        LLAMA_SELF_DISTURBANCE_FAILURE_NONE = 0,
        LLAMA_SELF_DISTURBANCE_FAILURE_MALFORMED_ARGUMENTS = 1,
        LLAMA_SELF_DISTURBANCE_FAILURE_WRONG_TOOL_OR_METHOD = 2,
        LLAMA_SELF_DISTURBANCE_FAILURE_LOCAL_DISPATCH = 3,
        LLAMA_SELF_DISTURBANCE_FAILURE_DOWNSTREAM_SERVICE = 4,
        LLAMA_SELF_DISTURBANCE_FAILURE_UPSTREAM_DEPENDENCY = 5,
    };

    struct llama_self_appraisal_vector {
        float novelty;
        float goal_relevance;
        float progress_error;
        float progress_velocity_error;
        float expectation_violation;
        float controllability_deficit;
        float failure_severity;
        float social_deficit;
        float reciprocity_deficit;
        float unresolved_commitment;
        float effort_cost;
    };

    struct llama_self_disturbance_delta {
        float total_disturbance;
        float contradiction_delta;
        float uncertainty_delta;
        float goal_pressure_delta;
        float recovery_urgency_delta;
        float followup_continuation_delta;
        float social_relevance_delta;
        float loop_inefficiency_delta;
        float satisfaction_risk_delta;
        uint32_t reason_mask;
        bool requires_emotive_recompute;
    };

    struct llama_self_disturbance_state_info {
        bool valid;
        int32_t source_kind;
        int32_t failure_class;
        float source_reliability;
        int64_t admitted_monotonic_ms;
        struct llama_self_appraisal_vector appraisal;
        struct llama_self_disturbance_delta delta;
    };

    enum llama_self_profile_id {
        LLAMA_SELF_PROFILE_GOAL_PROGRESS = 0,
        LLAMA_SELF_PROFILE_USER_OUTCOME,
        LLAMA_SELF_PROFILE_EPISTEMIC,
        LLAMA_SELF_PROFILE_EFFICIENCY,
        LLAMA_SELF_PROFILE_RECOVERY,
        LLAMA_SELF_PROFILE_STRATEGY,
        LLAMA_SELF_PROFILE_SELF_IMPROVEMENT,
    };

    enum llama_self_horizon_id {
        LLAMA_SELF_HORIZON_INSTANT = 0,
        LLAMA_SELF_HORIZON_SHORT = 1,
        LLAMA_SELF_HORIZON_LONG = 2,
        LLAMA_SELF_HORIZON_COUNT = 3,
    };

    struct llama_self_goal_progress_profile {
        float goal_progress_estimate;
        float blocker_severity;
        float dependency_readiness;
        float urgency;
        float expected_next_action_gain;
        float commitment_slippage_risk;
        float confidence;
    };

    struct llama_self_user_outcome_profile {
        float satisfaction_estimate;
        float frustration_risk;
        float misunderstanding_risk;
        float trust_repair_need;
        float preference_uncertainty;
        float cognitive_load_estimate;
        float autonomy_tolerance_estimate;
        float confidence;
    };

    struct llama_self_user_preference_profile {
        float directness_preference;
        float verbosity_preference;
        float structure_preference;
        float clarification_preference;
        float autonomy_preference;
        float disagreement_sensitivity;
        float rhetorical_intensity;
        float preference_confidence;
        float rhetorical_confidence;
        float simulator_readiness;
    };

    struct llama_self_epistemic_profile {
        float answerability;
        float evidence_sufficiency;
        float ambiguity_concentration;
        float self_estimate_confidence;
        float tool_need_confidence;
        float contradiction_load;
        float uncertainty_load;
    };

    struct llama_self_efficiency_profile {
        float expected_steps_remaining;
        float expected_inference_cost_remaining;
        float loop_inefficiency;
        float repetition_risk;
        float context_thrash_risk;
        float tool_roundtrip_cost;
        float response_compaction_opportunity;
    };

    struct llama_self_recovery_profile {
        float favorable_divergence_goal;
        float favorable_divergence_social;
        float favorable_divergence_epistemic;
        float favorable_divergence_action;
        float recovery_momentum;
        float regulation_debt;
        float unresolved_tension_load;
        float recovery_cost_estimate;
    };

    struct llama_self_strategy_profile {
        float answer_bias;
        float ask_bias;
        float act_bias;
        float wait_bias;
        float exploit_bias;
        float deliberate_bias;
        float write_internal_bias;
        float act_external_bias;
    };

    struct llama_self_improvement_profile {
        float update_worthiness;
        float expected_gain;
        float evidence_deficit;
        float reversibility;
        float blast_radius_risk;
        float observability_deficit;
        float readiness;
    };

    enum {
        LLAMA_SELF_MODEL_EXTENSION_KEY_MAX_CHARS = 96,
        LLAMA_SELF_MODEL_EXTENSION_LABEL_MAX_CHARS = 128,
        LLAMA_SELF_MODEL_EXTENSION_CONTENT_MAX_CHARS = 256,
    };

    struct llama_self_model_extension_update {
        int32_t source;
        int32_t source_tool_kind;
        int32_t kind;
        int32_t domain;
        int32_t lifecycle_stage;
        uint32_t flags;
        uint32_t support_count;
        float value;
        float desired_value;
        float desired_value_min;
        float desired_value_max;
        float confidence;
        float salience;
        float gain_weight;
        float allostatic_weight;
        float surprise_score;
        float relevance_score;
        float admission_score;
        float permanence_score;
        float stability_score;
        float allostatic_eligibility;
        char key[LLAMA_SELF_MODEL_EXTENSION_KEY_MAX_CHARS];
        char label[LLAMA_SELF_MODEL_EXTENSION_LABEL_MAX_CHARS];
        char content[LLAMA_SELF_MODEL_EXTENSION_CONTENT_MAX_CHARS];
    };

    struct llama_self_model_extension_info {
        int32_t slot;
        int32_t source;
        int32_t source_tool_kind;
        int32_t kind;
        int32_t domain;
        int32_t lifecycle_stage;
        uint32_t flags;
        int64_t last_update_monotonic_ms;
        uint32_t activation_count;
        uint32_t support_count;
        float value;
        float desired_value;
        float desired_value_min;
        float desired_value_max;
        float confidence;
        float salience;
        float gain_weight;
        float allostatic_weight;
        float surprise_score;
        float relevance_score;
        float admission_score;
        float permanence_score;
        float stability_score;
        float allostatic_eligibility;
        char key[LLAMA_SELF_MODEL_EXTENSION_KEY_MAX_CHARS];
        char label[LLAMA_SELF_MODEL_EXTENSION_LABEL_MAX_CHARS];
        char content[LLAMA_SELF_MODEL_EXTENSION_CONTENT_MAX_CHARS];
    };

    struct llama_self_model_extension_summary {
        int32_t active_count;
        int32_t transient_count;
        int32_t permanent_count;
        int32_t discovered_count;
        int32_t gain_count;
        int32_t allostatic_count;
        int32_t hard_memory_count;
        int32_t tool_count;
        float mean_confidence;
        float mean_salience;
        float mean_admission;
        float mean_permanence;
        float mean_allostatic_eligibility;
        float max_salience;
        float gain_signal;
        float gain_signal_abs;
        float context_activation;
        float allostatic_divergence;
    };

    enum {
        LLAMA_SELF_MODEL_EXTENSION_MAX_CANDIDATES = 4,
    };

    struct llama_self_model_extension_candidate {
        bool promoted;
        int32_t source_tool_kind;
        int32_t domain;
        float expected_gain_improvement;
        float expected_allostatic_delta;
        float confidence;
        char key[LLAMA_SELF_MODEL_EXTENSION_KEY_MAX_CHARS];
        char label[LLAMA_SELF_MODEL_EXTENSION_LABEL_MAX_CHARS];
    };

    struct llama_self_model_extension_trace {
        bool valid;
        int32_t candidate_count;
        int32_t promoted_count;
        int32_t winner_index;
        struct llama_self_model_extension_candidate candidates[LLAMA_SELF_MODEL_EXTENSION_MAX_CANDIDATES];
    };

    struct llama_self_model_horizon_info {
        int32_t horizon_id;
        int64_t last_update_monotonic_ms;
        struct llama_self_goal_progress_profile goal_progress;
        struct llama_self_user_outcome_profile user_outcome;
        struct llama_self_user_preference_profile user_preference;
        struct llama_self_epistemic_profile epistemic;
        struct llama_self_efficiency_profile efficiency;
        struct llama_self_recovery_profile recovery;
        struct llama_self_strategy_profile strategy;
        struct llama_self_improvement_profile self_improvement;
    };

    struct llama_self_forecast_trace {
        bool valid;
        int64_t issued_monotonic_ms;
        float predicted_steps_remaining;
        float predicted_inference_cost_remaining;
        float predicted_satisfaction_delta;
        float predicted_recovery_delta;
        float predicted_goal_progress_delta;
        float confidence;
    };

    struct llama_self_prediction_error_trace {
        bool valid;
        int64_t observed_after_monotonic_ms;
        float steps_error;
        float inference_cost_error;
        float satisfaction_error;
        float recovery_error;
        float goal_progress_error;
    };

    enum {
        LLAMA_SELF_BELIEF_MAX_SLOTS = 4,
        LLAMA_SELF_BELIEF_MAX_PROMOTION_CANDIDATES = 4,
    };

    struct llama_self_belief_slot_info {
        float pressure;
        float confidence;
        float novelty_support;
        float memory_support;
        float forecast_error_support;
        int64_t last_update_monotonic_ms;
    };

    struct llama_self_belief_summary {
        bool valid;
        float known_care_uncertainty;
        float missing_observation_uncertainty;
        float unmodeled_care_uncertainty;
        float residual_allostatic_pressure;
        float promotion_readiness;
        float belief_entropy;
        float belief_confidence;
        float max_slot_pressure;
        float slot_pressure_mean;
    };

    struct llama_self_model_promotion_candidate {
        bool valid;
        int32_t slot_index;
        float support_score;
        float allostatic_relevance;
        float suggested_desired_value;
        float stability_score;
        char suggested_label[LLAMA_SELF_MODEL_EXTENSION_LABEL_MAX_CHARS];
    };

    struct llama_self_model_state_info {
        int32_t horizon_count;
        int32_t belief_slot_count;
        int32_t promotion_candidate_count;
        struct llama_self_model_horizon_info horizons[LLAMA_SELF_HORIZON_COUNT];
        struct llama_self_forecast_trace forecast;
        struct llama_self_prediction_error_trace prediction_error;
        struct llama_self_belief_summary belief_summary;
        struct llama_self_belief_slot_info belief_slots[LLAMA_SELF_BELIEF_MAX_SLOTS];
        struct llama_self_model_promotion_candidate promotion_candidates[LLAMA_SELF_BELIEF_MAX_PROMOTION_CANDIDATES];
        struct llama_self_model_extension_summary extension_summary;
        struct llama_self_model_extension_trace last_extension_trace;
    };

    struct llama_self_register_updater_rule {
        int32_t register_id;
        uint32_t phase_mask;
        float baseline;
        float rise_gain;
        float fall_gain;
        float baseline_pull;
        int32_t feature_ids[LLAMA_SELF_MAX_UPDATER_RULE_TERMS];
        float feature_weights[LLAMA_SELF_MAX_UPDATER_RULE_TERMS];
        int32_t source_register_ids[LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS];
        float source_register_weights[LLAMA_SELF_MAX_UPDATER_RULE_SOURCE_REGISTERS];
    };

    struct llama_self_updater_program {
        uint32_t version;
        float memory_novelty_weight;
        float memory_working_similarity_weight;
        float memory_handle_similarity_weight;
        float memory_uncertainty_weight;
        float memory_contradiction_weight;
        float memory_handle_variance_weight;
        float broadcast_social_weight;
        float broadcast_contradiction_weight;
        float broadcast_uncertainty_weight;
        float broadcast_tool_pending_weight;
        float broadcast_tool_unready_weight;
        float broadcast_failure_weight;
        float broadcast_question_weight;
        float broadcast_goal_weight;
        float repair_emit_threshold;
        float repair_dissatisfaction_floor;
        float repair_recent_user_valence_floor;
        float repair_inhibition_max;
        float repair_admission_floor;
        float repair_admission_weight;
        uint32_t rule_count;
        struct llama_self_register_updater_rule rules[LLAMA_SELF_MAX_UPDATER_RULES];
    };

    struct llama_self_counterfactual_result {
        uint32_t updater_version;
        int32_t replay_channel;
        int32_t replayed_events;
        int32_t working_memory_count;
        int32_t reactivation_count;
        float uncertainty;
        float contradiction;
        float memory_write_priority;
        float broadcast_pressure;
    };

    enum llama_active_loop_action {
        LLAMA_ACTIVE_LOOP_ACTION_ANSWER = 0,
        LLAMA_ACTIVE_LOOP_ACTION_ASK    = 1,
        LLAMA_ACTIVE_LOOP_ACTION_ACT    = 2,
        LLAMA_ACTIVE_LOOP_ACTION_WAIT   = 3,
    };

    enum llama_dmn_action {
        LLAMA_DMN_ACTION_SILENT         = 0,
        LLAMA_DMN_ACTION_INTERNAL_WRITE = 1,
        LLAMA_DMN_ACTION_INVOKE_TOOL    = 2,
        LLAMA_DMN_ACTION_EMIT           = 3,
    };

    enum llama_cognitive_loop_phase {
        LLAMA_COG_LOOP_PHASE_ASSEMBLE     = 0,
        LLAMA_COG_LOOP_PHASE_PROPOSE      = 1,
        LLAMA_COG_LOOP_PHASE_PREPARE_TOOL = 2,
        LLAMA_COG_LOOP_PHASE_WAIT_TOOL    = 3,
        LLAMA_COG_LOOP_PHASE_OBSERVE      = 4,
        LLAMA_COG_LOOP_PHASE_FINISH       = 5,
    };

    enum llama_cognitive_terminal_reason {
        LLAMA_COG_TERMINAL_NONE                  = 0,
        LLAMA_COG_TERMINAL_ANSWER_READY          = 1,
        LLAMA_COG_TERMINAL_ASK_USER              = 2,
        LLAMA_COG_TERMINAL_TOOL_REQUIRED         = 3,
        LLAMA_COG_TERMINAL_WAITING_ON_TOOL       = 4,
        LLAMA_COG_TERMINAL_BACKGROUND_DEFERRED   = 5,
        LLAMA_COG_TERMINAL_PRESSURE_NOT_ADMITTED = 6,
        LLAMA_COG_TERMINAL_INTERNAL_WRITE_READY  = 7,
        LLAMA_COG_TERMINAL_EMIT_READY            = 8,
        LLAMA_COG_TERMINAL_GOVERNANCE_BLOCKED    = 9,
    };

    enum llama_cognitive_plan_mode {
        LLAMA_COG_PLAN_MODE_NONE = 0,
        LLAMA_COG_PLAN_MODE_COMPOSITION = 1,
    };

    enum llama_cognitive_plan_status {
        LLAMA_COG_PLAN_STATUS_NONE = 0,
        LLAMA_COG_PLAN_STATUS_DRAFT = 1,
        LLAMA_COG_PLAN_STATUS_EXECUTING = 2,
        LLAMA_COG_PLAN_STATUS_WAITING_TOOL = 3,
        LLAMA_COG_PLAN_STATUS_COMPLETED = 4,
        LLAMA_COG_PLAN_STATUS_BLOCKED = 5,
    };

    enum llama_cognitive_plan_step_kind {
        LLAMA_COG_PLAN_STEP_NONE = 0,
        LLAMA_COG_PLAN_STEP_EMIT_ANSWER = 1,
        LLAMA_COG_PLAN_STEP_EMIT_ASK = 2,
        LLAMA_COG_PLAN_STEP_INVOKE_TOOL = 3,
        LLAMA_COG_PLAN_STEP_OBSERVE_TOOL = 4,
        LLAMA_COG_PLAN_STEP_INTERNAL_WRITE = 5,
        LLAMA_COG_PLAN_STEP_EMIT_BACKGROUND = 6,
        LLAMA_COG_PLAN_STEP_WAIT = 7,
    };

    enum llama_cognitive_plan_step_status {
        LLAMA_COG_PLAN_STEP_STATUS_NONE = 0,
        LLAMA_COG_PLAN_STEP_STATUS_PENDING = 1,
        LLAMA_COG_PLAN_STEP_STATUS_READY = 2,
        LLAMA_COG_PLAN_STEP_STATUS_ACTIVE = 3,
        LLAMA_COG_PLAN_STEP_STATUS_COMPLETED = 4,
        LLAMA_COG_PLAN_STEP_STATUS_BLOCKED = 5,
        LLAMA_COG_PLAN_STEP_STATUS_SKIPPED = 6,
    };

    enum llama_cognitive_tool_flags {
        LLAMA_COG_TOOL_ACTIVE_ELIGIBLE      = 1u << 0,
        LLAMA_COG_TOOL_DMN_ELIGIBLE         = 1u << 1,
        LLAMA_COG_TOOL_SIMULATION_SAFE      = 1u << 2,
        LLAMA_COG_TOOL_REMEDIATION_SAFE     = 1u << 3,
        LLAMA_COG_TOOL_EXTERNAL_SIDE_EFFECT = 1u << 4,
    };

    enum llama_cognitive_tool_latency_class {
        LLAMA_COG_TOOL_LATENCY_LOW    = 0,
        LLAMA_COG_TOOL_LATENCY_MEDIUM = 1,
        LLAMA_COG_TOOL_LATENCY_HIGH   = 2,
    };

    enum llama_cognitive_command_kind {
        LLAMA_COG_COMMAND_NONE            = 0,
        LLAMA_COG_COMMAND_EMIT_ANSWER     = 1,
        LLAMA_COG_COMMAND_EMIT_ASK        = 2,
        LLAMA_COG_COMMAND_INVOKE_TOOL     = 3,
        LLAMA_COG_COMMAND_INTERNAL_WRITE  = 4,
        LLAMA_COG_COMMAND_EMIT_BACKGROUND = 5,
    };

    enum llama_cognitive_command_origin {
        LLAMA_COG_COMMAND_ORIGIN_ACTIVE = 0,
        LLAMA_COG_COMMAND_ORIGIN_DMN    = 1,
    };

    enum llama_cognitive_command_status {
        LLAMA_COG_COMMAND_STATUS_PENDING   = 0,
        LLAMA_COG_COMMAND_STATUS_ACKED     = 1,
        LLAMA_COG_COMMAND_STATUS_COMPLETED = 2,
        LLAMA_COG_COMMAND_STATUS_CANCELLED = 3,
    };

    enum llama_cognitive_reason_flags {
        LLAMA_COG_REASON_ROLE_TOOL           = 1u << 0,
        LLAMA_COG_REASON_TOOL_COMPLETED      = 1u << 1,
        LLAMA_COG_REASON_QUESTION_SIGNAL     = 1u << 2,
        LLAMA_COG_REASON_TOOL_AFFORDANCE     = 1u << 3,
        LLAMA_COG_REASON_HIGH_UNCERTAINTY    = 1u << 4,
        LLAMA_COG_REASON_HIGH_INHIBITION     = 1u << 5,
        LLAMA_COG_REASON_HIGH_CONTINUATION   = 1u << 6,
        LLAMA_COG_REASON_REACTIVATION_TARGET = 1u << 7,
        LLAMA_COG_REASON_SOCIAL_RELEVANCE    = 1u << 8,
        LLAMA_COG_REASON_PRESSURE_THRESHOLD  = 1u << 9,
    };

    enum llama_dmn_seed_source_flags {
        LLAMA_DMN_SEED_SOURCE_REGISTERS    = 1u << 0,
        LLAMA_DMN_SEED_SOURCE_REACTIVATION = 1u << 1,
        LLAMA_DMN_SEED_SOURCE_SELF_STATE   = 1u << 2,
        LLAMA_DMN_SEED_SOURCE_TOOL_STATE   = 1u << 3,
        LLAMA_DMN_SEED_SOURCE_WORKING_MEM  = 1u << 4,
    };

    enum llama_dmn_maintenance_flags {
        LLAMA_DMN_MAINTENANCE_COMPRESS_WORKING_MEMORY = 1u << 0,
        LLAMA_DMN_MAINTENANCE_PAST_LORA_TICK          = 1u << 1,
        LLAMA_DMN_MAINTENANCE_REFRESH_REACTIVATION    = 1u << 2,
    };

    enum llama_favorable_dimension_id {
        LLAMA_FAVORABLE_DIM_CONTRADICTION = 0,
        LLAMA_FAVORABLE_DIM_UNCERTAINTY,
        LLAMA_FAVORABLE_DIM_MEMORY_WRITE_PRIORITY,
        LLAMA_FAVORABLE_DIM_REACTIVATION_PRIORITY,
        LLAMA_FAVORABLE_DIM_TOOL_BACKLOG,
        LLAMA_FAVORABLE_DIM_TOOL_READINESS,
        LLAMA_FAVORABLE_DIM_SOCIAL_TRUST,
        LLAMA_FAVORABLE_DIM_SOCIAL_RECIPROCITY,
        LLAMA_FAVORABLE_DIM_BROADCAST_PRESSURE,
        LLAMA_FAVORABLE_DIM_BROADCAST_INHIBITION,
        LLAMA_FAVORABLE_DIM_FOLLOWUP_CONTINUATION,
        LLAMA_FAVORABLE_DIM_SOCIAL_DISSATISFACTION,
    };

    enum llama_counterfactual_risk_tier {
        LLAMA_COUNTERFACTUAL_RISK_LOW = 0,
        LLAMA_COUNTERFACTUAL_RISK_MEDIUM = 1,
        LLAMA_COUNTERFACTUAL_RISK_HIGH = 2,
    };

    enum llama_counterfactual_family {
        LLAMA_COUNTERFACTUAL_FAMILY_MESSAGE_VARIANT = 0,
        LLAMA_COUNTERFACTUAL_FAMILY_TOOL_ARGUMENTS  = 1,
        LLAMA_COUNTERFACTUAL_FAMILY_HARD_MEMORY_QUERY = 2,
        LLAMA_COUNTERFACTUAL_FAMILY_TOOL_CHOICE     = 3,
        LLAMA_COUNTERFACTUAL_FAMILY_TIMING_SHIFT    = 4,
        LLAMA_COUNTERFACTUAL_FAMILY_LORA_ABLATION   = 5,
        LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_LOCAL = 6,
        LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_HISTORY = 7,
        LLAMA_COUNTERFACTUAL_FAMILY_FUNCTIONAL_ORTHOGONAL = 8,
        LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_LOCAL = 9,
        LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_HISTORY = 10,
        LLAMA_COUNTERFACTUAL_FAMILY_PROCESS_FUNCTIONAL_ORTHOGONAL = 11,
        LLAMA_COUNTERFACTUAL_FAMILY_SENSITIVITY     = 12,
        LLAMA_COUNTERFACTUAL_FAMILY_UPDATER_POLICY  = 13,
    };

    enum llama_remediation_action {
        LLAMA_REMEDIATION_ACTION_NONE = 0,
        LLAMA_REMEDIATION_ACTION_GATHER_INFO = 1,
        LLAMA_REMEDIATION_ACTION_ACTIVE_LORA_UPDATE = 2,
    };

    enum llama_governance_outcome {
        LLAMA_GOVERNANCE_OUTCOME_ALLOW = 0,
        LLAMA_GOVERNANCE_OUTCOME_DENY = 1,
        LLAMA_GOVERNANCE_OUTCOME_DEFER = 2,
        LLAMA_GOVERNANCE_OUTCOME_EMIT_REPAIR = 3,
    };

    enum {
        LLAMA_FUNCTIONAL_MAX_SNAPSHOTS_PER_FAMILY = 4,
        LLAMA_ACTIVE_LOOP_MAX_CANDIDATES = 4,
        LLAMA_DMN_MAX_CANDIDATES = 4,
        LLAMA_DMN_MAX_REACTIVATION_TARGETS = 4,
        LLAMA_DMN_SEED_DIMS = 8,
        LLAMA_FAVORABLE_MAX_DIMS = 12,
        LLAMA_COUNTERFACTUAL_MAX_CANDIDATES = 12,
        LLAMA_REPAIR_MESSAGE_MAX_CHARS = 256,
        LLAMA_HARD_MEMORY_MAX_RESULTS = 8,
        LLAMA_HARD_MEMORY_MAX_TEXT_CHARS = 256,
        LLAMA_HARD_MEMORY_MAX_TITLE_CHARS = 128,
        LLAMA_HARD_MEMORY_MAX_ID_CHARS = 96,
        LLAMA_HARD_MEMORY_MAX_TAG_CHARS = 128,
        LLAMA_HARD_MEMORY_MAX_URL_CHARS = 256,
        LLAMA_HARD_MEMORY_MAX_TOKEN_CHARS = 256,
        LLAMA_HARD_MEMORY_MAX_ERROR_CHARS = 256,
        LLAMA_HARD_MEMORY_MAX_PROFILE_CHARS = 512,
        LLAMA_HARD_MEMORY_QUERY_MAX_CHARS = 256,
        LLAMA_HARD_MEMORY_MAX_PRIMITIVES = 6,
        LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS = 4,
        LLAMA_BASH_TOOL_POLICY_MAX_CHARS = 512,
        LLAMA_BASH_TOOL_PATH_MAX_CHARS = 256,
        LLAMA_BASH_TOOL_CWD_MAX_CHARS = 256,
        LLAMA_BASH_TOOL_COMMAND_MAX_CHARS = 512,
        LLAMA_BASH_TOOL_INTENT_MAX_CHARS = 512,
        LLAMA_BASH_TOOL_STDOUT_MAX_CHARS = 819200,
        LLAMA_BASH_TOOL_STDERR_MAX_CHARS = 4096,
        LLAMA_BASH_TOOL_ERROR_MAX_CHARS = 256,
        LLAMA_CODEX_TOOL_PATH_MAX_CHARS = 256,
        LLAMA_CODEX_TOOL_CWD_MAX_CHARS = 256,
        LLAMA_CODEX_TOOL_PROMPT_MAX_CHARS = 4096,
        LLAMA_CODEX_TOOL_SUMMARY_MAX_CHARS = 2048,
        LLAMA_CODEX_TOOL_MANUAL_MAX_CHARS = 1024,
        LLAMA_CODEX_TOOL_FILES_MAX_CHARS = 2048,
        LLAMA_CODEX_TOOL_STDOUT_MAX_CHARS = 8192,
        LLAMA_CODEX_TOOL_STDERR_MAX_CHARS = 8192,
        LLAMA_CODEX_TOOL_ERROR_MAX_CHARS = 512,
        LLAMA_DMN_MAX_REPORTABLE_CONCEPTS = 4,
        LLAMA_DMN_PROMPT_MAX_CHARS = 512,
        LLAMA_DMN_PROMPT_OUTLINE_MAX_CHARS = 256,
        LLAMA_DMN_REASONING_MAX_CHARS = 1024,
        LLAMA_TELEGRAM_RELAY_TEXT_MAX_CHARS = 1024,
        LLAMA_TELEGRAM_RELAY_DEDUPE_MAX_CHARS = 128,
        LLAMA_TELEGRAM_RELAY_ERROR_MAX_CHARS = 256,
        LLAMA_TELEGRAM_CHAT_SCOPE_MAX_CHARS = 64,
        LLAMA_TELEGRAM_ASK_QUESTION_MAX_CHARS = 512,
        LLAMA_TELEGRAM_ASK_OPTION_LABEL_MAX_CHARS = 96,
        LLAMA_TELEGRAM_ASK_MAX_OPTIONS = 6,
        LLAMA_SELF_STATE_MAX_DELTA_DIMS = 8,
        LLAMA_COGNITIVE_MAX_TOOL_SPECS = 64,
        LLAMA_COGNITIVE_TOOL_NAME_MAX_CHARS = 48,
        LLAMA_COGNITIVE_TOOL_DESCRIPTION_MAX_CHARS = 160,
        LLAMA_COGNITIVE_MAX_PENDING_COMMANDS = 16,
        LLAMA_COGNITIVE_MAX_PLAN_STEPS = 5,
        LLAMA_SELF_MODEL_EXTENSION_MAX_ITEMS = 32,
        LLAMA_PROCESS_FUNCTIONAL_MAX_ENTRIES = 16,
        LLAMA_PROCESS_FUNCTIONAL_KEY_MAX_CHARS = 160,
        LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS = 96,
        LLAMA_OPENCLAW_PLUGIN_ID_MAX_CHARS = 64,
        LLAMA_OPENCLAW_NAMESPACE_MAX_CHARS = 128,
    };

    enum llama_process_functional_scope_kind {
        LLAMA_PROCESS_FUNCTIONAL_SCOPE_PROCESS = 0,
        LLAMA_PROCESS_FUNCTIONAL_SCOPE_PROCESS_STEP = 1,
    };

    enum llama_process_functional_creation_reason {
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_NONE = 0,
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_THRESHOLD_NOT_MET = 1,
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_CREATED = 2,
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_EXISTING_ENTRY = 3,
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_COOLDOWN = 4,
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_CAPACITY_DENIED = 5,
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_EVICTED_REPLACEMENT = 6,
        LLAMA_PROCESS_FUNCTIONAL_CREATION_REASON_INVALID_SIGNATURE = 7,
    };

    struct llama_process_functional_params {
        bool enabled;
        uint32_t max_entries;
        uint32_t min_observations;
        float noop_abs_ceiling;
        float weak_positive_ceiling;
        float mean_outcome_ceiling;
        float weak_or_worse_ratio_threshold;
        uint32_t creation_cooldown_updates;
        float utility_decay;
    };

    struct llama_process_functional_signature {
        bool valid;
        uint64_t signature_hash;
        int32_t scope_kind;
        int32_t family;
        int32_t loop_origin;
        int32_t microphase;
        int32_t plan_mode;
        int32_t plan_step_kind;
        int32_t tool_kind;
        int32_t source_family;
        bool requires_tool_result;
        int32_t transient_plan_id;
        int32_t transient_step_index;
        int32_t transient_source_id;
        char tool_name[LLAMA_COGNITIVE_TOOL_NAME_MAX_CHARS];
        char capability_id[LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS];
        char provenance_namespace[LLAMA_OPENCLAW_NAMESPACE_MAX_CHARS];
        char semantic_key[LLAMA_PROCESS_FUNCTIONAL_KEY_MAX_CHARS];
    };

    struct llama_process_functional_ledger_info {
        bool valid;
        struct llama_process_functional_signature signature;
        uint32_t observation_count;
        uint32_t negative_count;
        uint32_t noop_count;
        uint32_t weak_positive_count;
        uint32_t strong_positive_count;
        float mean_signed_outcome;
        float ema_signed_outcome;
        float mean_magnitude;
        float weak_or_worse_ratio;
        uint32_t creation_attempt_count;
        int32_t last_creation_reason;
        uint64_t last_observed_us;
        uint64_t last_creation_attempt_us;
    };

    struct llama_process_functional_entry_info {
        bool valid;
        int32_t slot;
        struct llama_process_functional_signature signature;
        uint64_t created_at_us;
        uint64_t last_used_us;
        uint64_t activation_count;
        uint64_t update_count;
        float utility_score;
        float current_gain;
        float last_signed_outcome;
        float current_bootstrap_std;
        float last_bootstrap_perturbation;
    };

    struct llama_process_functional_trace {
        bool valid;
        struct llama_process_functional_signature signature;
        bool matched_existing_entry;
        int32_t matched_entry_slot;
        bool created_entry;
        int32_t created_entry_slot;
        int32_t creation_reason;
        int32_t evicted_entry_slot;
        float signed_outcome;
        float magnitude;
        float weak_or_worse_ratio;
        uint32_t bank_size;
        uint32_t bank_capacity;
        bool activation_attached;
    };

    struct llama_hard_memory_config {
        bool enabled;
        bool archive_enabled;
        bool include_profile_by_default;
        bool archive_counterfactual_events;
        int32_t timeout_ms;
        int32_t max_results;
        float query_threshold;
        float archival_delta_threshold;
        char base_url[LLAMA_HARD_MEMORY_MAX_URL_CHARS];
        char auth_token[LLAMA_HARD_MEMORY_MAX_TOKEN_CHARS];
        char container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
        char runtime_identity[LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
    };

    enum llama_hard_memory_primitive_kind {
        LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT      = 0,
        LLAMA_HARD_MEMORY_PRIMITIVE_TRAJECTORY          = 1,
        LLAMA_HARD_MEMORY_PRIMITIVE_OUTCOME             = 2,
        LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_OBSERVATION    = 3,
        LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL          = 4,
        LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT = 5,
    };

    enum llama_hard_memory_domain {
        LLAMA_HARD_MEMORY_DOMAIN_GOAL_PROGRESS    = 0,
        LLAMA_HARD_MEMORY_DOMAIN_USER_OUTCOME     = 1,
        LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC        = 2,
        LLAMA_HARD_MEMORY_DOMAIN_EFFICIENCY       = 3,
        LLAMA_HARD_MEMORY_DOMAIN_RECOVERY         = 4,
        LLAMA_HARD_MEMORY_DOMAIN_STRATEGY         = 5,
        LLAMA_HARD_MEMORY_DOMAIN_SELF_IMPROVEMENT = 6,
    };

    enum llama_hard_memory_primitive_flags {
        LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_GAIN       = 1u << 0,
        LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_ALLOSTASIS = 1u << 1,
        LLAMA_HARD_MEMORY_PRIMITIVE_VALIDATED         = 1u << 2,
        LLAMA_HARD_MEMORY_PRIMITIVE_TOOL_DERIVED      = 1u << 3,
    };

    struct llama_hard_memory_primitive {
        int32_t kind;
        int32_t domain;
        int32_t source_role;
        int32_t source_channel;
        int32_t source_tool_kind;
        int32_t transaction_id;
        uint32_t flags;
        float importance;
        float confidence;
        float gain_bias;
        float allostatic_relevance;
        char key[LLAMA_HARD_MEMORY_MAX_ID_CHARS];
        char title[LLAMA_HARD_MEMORY_MAX_TITLE_CHARS];
        char content[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS];
        char tags[LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS][LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
    };

    struct llama_hard_memory_primitive_summary {
        int32_t kind;
        int32_t domain;
        int32_t source_tool_kind;
        uint32_t flags;
        float importance;
        float confidence;
        float gain_bias;
        float allostatic_relevance;
        char key[LLAMA_HARD_MEMORY_MAX_ID_CHARS];
        char title[LLAMA_HARD_MEMORY_MAX_TITLE_CHARS];
    };

    struct llama_hard_memory_write_item {
        bool is_static;
        struct llama_hard_memory_primitive primitive;
    };

    struct llama_hard_memory_retrieval_summary {
        int32_t event_count;
        int32_t trajectory_count;
        int32_t outcome_count;
        int32_t tool_observation_count;
        int32_t user_model_count;
        int32_t self_model_count;
        float mean_similarity;
        float max_similarity;
        float importance_signal;
        float confidence_signal;
        float gain_support;
        float allostatic_support;
        float goal_support;
        float user_support;
        float epistemic_support;
        float efficiency_support;
        float recovery_support;
        float strategy_support;
        float self_improvement_support;
        float user_preference_support;
        float user_rhetorical_support;
    };

    struct llama_hard_memory_query_request {
        int32_t limit;
        float threshold;
        bool include_profile;
        bool use_temporal_self_hint;
        int32_t temporal_adapter_role;
        char query[LLAMA_HARD_MEMORY_QUERY_MAX_CHARS];
        char container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
    };

    struct llama_hard_memory_hit {
        bool memory_result;
        float similarity;
        int32_t kind;
        int32_t domain;
        int32_t source_role;
        int32_t source_channel;
        int32_t source_tool_kind;
        uint32_t flags;
        float importance;
        float confidence;
        float gain_bias;
        float allostatic_relevance;
        char id[LLAMA_HARD_MEMORY_MAX_ID_CHARS];
        char title[LLAMA_HARD_MEMORY_MAX_TITLE_CHARS];
        char content[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS];
        char tags[LLAMA_HARD_MEMORY_MAX_PRIMITIVE_TAGS][LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
    };

    struct llama_self_state_delta_dimension {
        int32_t register_id;
        float before_value;
        float after_value;
        float abs_delta;
    };

    struct llama_self_state_delta_summary {
        float total_delta;
        float max_delta;
        int32_t dimension_count;
        int32_t role;
        int32_t channel;
        uint32_t flags;
        struct llama_self_state_delta_dimension dimensions[LLAMA_SELF_STATE_MAX_DELTA_DIMS];
    };

    struct llama_hard_memory_result {
        bool ok;
        bool profile_included;
        int32_t tool_kind;
        int32_t status_code;
        int32_t result_count;
        int64_t request_started_us;
        int64_t request_completed_us;
        char effective_container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
        char profile_static[LLAMA_HARD_MEMORY_MAX_PROFILE_CHARS];
        char profile_dynamic[LLAMA_HARD_MEMORY_MAX_PROFILE_CHARS];
        char error[LLAMA_HARD_MEMORY_MAX_ERROR_CHARS];
        struct llama_hard_memory_retrieval_summary retrieval_summary;
        struct llama_hard_memory_hit results[LLAMA_HARD_MEMORY_MAX_RESULTS];
    };

    struct llama_hard_memory_archive_trace {
        bool attempted;
        bool archived;
        int32_t tool_kind;
        int32_t status_code;
        int32_t primitive_count;
        int64_t request_started_us;
        int64_t request_completed_us;
        char custom_id[LLAMA_HARD_MEMORY_MAX_ID_CHARS];
        char container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
        char content_excerpt[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS];
        char error[LLAMA_HARD_MEMORY_MAX_ERROR_CHARS];
        struct llama_self_state_delta_summary delta;
        struct llama_hard_memory_primitive_summary primitives[LLAMA_HARD_MEMORY_MAX_PRIMITIVES];
    };

    struct llama_bash_tool_config {
        bool enabled;
        bool inherit_env;
        bool login_shell;
        bool reject_shell_metacharacters;
        int32_t timeout_ms;
        int32_t cpu_time_limit_secs;
        int32_t max_child_processes;
        int32_t max_open_files;
        int32_t max_file_size_bytes;
        int32_t max_stdout_bytes;
        int32_t max_stderr_bytes;
        char bash_path[LLAMA_BASH_TOOL_PATH_MAX_CHARS];
        char working_directory[LLAMA_BASH_TOOL_CWD_MAX_CHARS];
        char allowed_commands[LLAMA_BASH_TOOL_POLICY_MAX_CHARS];
        char blocked_patterns[LLAMA_BASH_TOOL_POLICY_MAX_CHARS];
        char allowed_env[LLAMA_BASH_TOOL_POLICY_MAX_CHARS];
    };

    struct llama_bash_tool_request {
        int32_t command_id;
        int32_t origin;
        int32_t tool_job_id;
        int32_t timeout_ms;
        int32_t cpu_time_limit_secs;
        int32_t max_child_processes;
        int32_t max_open_files;
        int32_t max_file_size_bytes;
        int32_t max_stdout_bytes;
        int32_t max_stderr_bytes;
        bool inherit_env;
        bool login_shell;
        bool reject_shell_metacharacters;
        bool command_ready;
        char bash_path[LLAMA_BASH_TOOL_PATH_MAX_CHARS];
        char working_directory[LLAMA_BASH_TOOL_CWD_MAX_CHARS];
        char allowed_commands[LLAMA_BASH_TOOL_POLICY_MAX_CHARS];
        char blocked_patterns[LLAMA_BASH_TOOL_POLICY_MAX_CHARS];
        char allowed_env[LLAMA_BASH_TOOL_POLICY_MAX_CHARS];
        char intent_text[LLAMA_BASH_TOOL_INTENT_MAX_CHARS];
        char command_text[LLAMA_BASH_TOOL_COMMAND_MAX_CHARS];
    };

    struct llama_bash_tool_result {
        int32_t command_id;
        int32_t tool_job_id;
        int32_t exit_code;
        int32_t term_signal;
        int32_t runtime_ms;
        bool timed_out;
        bool launch_failed;
        bool truncated_stdout;
        bool truncated_stderr;
        char stdout_text[LLAMA_BASH_TOOL_STDOUT_MAX_CHARS];
        char stderr_text[LLAMA_BASH_TOOL_STDERR_MAX_CHARS];
        char error_text[LLAMA_BASH_TOOL_ERROR_MAX_CHARS];
    };

    struct llama_codex_tool_config {
        bool enabled;
        bool dangerous_no_approval;
        bool rebuild_after_changes;
        bool verify_tool_access_after_rebuild;
        int32_t timeout_ms;
        int32_t max_stdout_bytes;
        int32_t max_stderr_bytes;
        char codex_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
        char working_directory[LLAMA_CODEX_TOOL_CWD_MAX_CHARS];
        char rebuild_script_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
        char rebuild_helper_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
        char completion_message_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
    };

    struct llama_codex_tool_request {
        int32_t command_id;
        int32_t origin;
        int32_t tool_job_id;
        int32_t timeout_ms;
        int32_t max_stdout_bytes;
        int32_t max_stderr_bytes;
        bool dangerous_no_approval;
        bool rebuild_after_changes;
        bool verify_tool_access_after_rebuild;
        bool command_ready;
        char codex_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
        char working_directory[LLAMA_CODEX_TOOL_CWD_MAX_CHARS];
        char rebuild_script_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
        char rebuild_helper_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
        char completion_message_path[LLAMA_CODEX_TOOL_PATH_MAX_CHARS];
        char intent_text[LLAMA_CODEX_TOOL_PROMPT_MAX_CHARS];
        char task_prompt[LLAMA_CODEX_TOOL_PROMPT_MAX_CHARS];
    };

    struct llama_codex_tool_result {
        int32_t command_id;
        int32_t tool_job_id;
        int32_t exit_code;
        int32_t runtime_ms;
        bool launch_failed;
        bool repo_changed;
        bool rebuild_attempted;
        bool rebuild_succeeded;
        bool accessibility_verified;
        bool truncated_stdout;
        bool truncated_stderr;
        char stdout_text[LLAMA_CODEX_TOOL_STDOUT_MAX_CHARS];
        char stderr_text[LLAMA_CODEX_TOOL_STDERR_MAX_CHARS];
        char error_text[LLAMA_CODEX_TOOL_ERROR_MAX_CHARS];
        char summary_text[LLAMA_CODEX_TOOL_SUMMARY_MAX_CHARS];
        char manual_requirements[LLAMA_CODEX_TOOL_MANUAL_MAX_CHARS];
        char changed_files_excerpt[LLAMA_CODEX_TOOL_FILES_MAX_CHARS];
    };

    enum llama_cognitive_hard_memory_operation {
        LLAMA_COG_HARD_MEMORY_OPERATION_QUERY = 0,
        LLAMA_COG_HARD_MEMORY_OPERATION_WRITE = 1,
    };

    enum llama_dmn_reportable_concept_kind {
        LLAMA_DMN_CONCEPT_NONE = 0,
        LLAMA_DMN_CONCEPT_MOTIVE = 1,
        LLAMA_DMN_CONCEPT_TENSION = 2,
        LLAMA_DMN_CONCEPT_QUESTION = 3,
        LLAMA_DMN_CONCEPT_NEXT_ACTION = 4,
    };

    enum llama_telegram_relay_intent_kind {
        LLAMA_TELEGRAM_RELAY_QUESTION = 0,
        LLAMA_TELEGRAM_RELAY_COMMENT = 1,
        LLAMA_TELEGRAM_RELAY_CONCLUSION = 2,
    };

    enum llama_authoritative_react_action {
        LLAMA_AUTHORITATIVE_REACT_ACTION_NONE = 0,
        LLAMA_AUTHORITATIVE_REACT_ACTION_ANSWER = 1,
        LLAMA_AUTHORITATIVE_REACT_ACTION_ASK = 2,
        LLAMA_AUTHORITATIVE_REACT_ACTION_ACT = 3,
        LLAMA_AUTHORITATIVE_REACT_ACTION_WAIT = 4,
        LLAMA_AUTHORITATIVE_REACT_ACTION_INTERNAL_WRITE = 5,
    };

    enum llama_authoritative_turn_status {
        LLAMA_AUTHORITATIVE_TURN_STATUS_DRAFTING        = 0,
        LLAMA_AUTHORITATIVE_TURN_STATUS_VALIDATING      = 1,
        LLAMA_AUTHORITATIVE_TURN_STATUS_WAITING_ON_TOOL = 2,
        LLAMA_AUTHORITATIVE_TURN_STATUS_COMPLETED       = 3,
        LLAMA_AUTHORITATIVE_TURN_STATUS_FAILED          = 4,
    };

    enum llama_authoritative_turn_validation_error_kind {
        LLAMA_AUTHORITATIVE_TURN_VALIDATION_NONE             = 0,
        LLAMA_AUTHORITATIVE_TURN_VALIDATION_PARSE_FAILED     = 1,
        LLAMA_AUTHORITATIVE_TURN_VALIDATION_BAD_ACTION       = 2,
        LLAMA_AUTHORITATIVE_TURN_VALIDATION_BAD_TOOL         = 3,
        LLAMA_AUTHORITATIVE_TURN_VALIDATION_BAD_TOOL_PAYLOAD = 4,
        LLAMA_AUTHORITATIVE_TURN_VALIDATION_POLICY_REJECTED  = 5,
    };

    struct llama_dmn_reportable_concept {
        bool valid;
        int32_t kind;
        float salience;
        float confidence;
        float user_contact_affordance;
        float tool_affordance;
        char hint[LLAMA_DMN_PROMPT_OUTLINE_MAX_CHARS];
    };

    struct llama_dmn_self_model_revision {
        bool valid;
        int32_t revision_id;
        uint64_t input_hash;
        float materiality_score;
        bool requires_prompt_regen;
    };

    struct llama_self_model_revision {
        bool valid;
        int32_t revision_id;
        uint64_t source_hash;
        uint64_t changed_register_mask;
        bool belief_changed;
        bool forecast_changed;
        bool social_changed;
        bool tool_changed;
        bool extension_changed;
        float allostatic_distance;
        float materiality_score;
        uint32_t materiality_reason_mask;
        bool requires_emotive_recompute;
    };

    struct llama_emotive_affective_vector {
        float valence;
        float arousal;
        float dominance;
        float social_closeness;
        float cognitive_tension;
        float goal_pressure;
    };

    struct llama_emotive_moment_revision {
        bool valid;
        int32_t revision_id;
        int32_t source_self_model_revision_id;
        uint64_t source_hash;
        struct llama_emotive_affective_vector affective_vector;
        int32_t lexical_profile_id;
        float materiality_score;
        uint64_t text_hash;
        char text[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS];
    };

    struct llama_authoritative_react_turn_state {
        bool valid;
        int32_t turn_id;
        int32_t origin;
        int32_t status;
        int64_t thought_context_item_id;
        int32_t action;
        int32_t selected_tool_kind;
        int32_t selected_tool_spec_index;
        int64_t tool_call_context_item_id;
        int64_t visible_output_context_item_id;
        int64_t observation_context_item_id;
        int32_t retry_count;
        int32_t validation_error_kind;
        int32_t source_emotive_revision_id;
        int64_t source_context_revision;
        int32_t tool_job_id;
        char selected_tool_name[LLAMA_COGNITIVE_TOOL_NAME_MAX_CHARS];
        char tool_call_payload[LLAMA_DMN_PROMPT_MAX_CHARS];
    };

    struct llama_learning_attribution_record {
        bool valid;
        int32_t attribution_id;
        int32_t turn_id;
        int64_t preceding_thought_context_item_id;
        int64_t selected_tool_context_item_id;
        int64_t tool_call_context_item_id;
        int64_t observation_context_item_id;
        int64_t post_observation_thought_context_item_id;
        float self_state_delta;
        float allostatic_distance_before;
        float allostatic_distance_after;
        float update_magnitude;
        float update_polarity;
    };

    struct llama_cognitive_hard_memory_request {
        int32_t command_id;
        int32_t origin;
        int32_t tool_job_id;
        int32_t operation;
        struct llama_hard_memory_query_request query;
        int32_t write_count;
        struct llama_hard_memory_write_item write_items[LLAMA_HARD_MEMORY_MAX_PRIMITIVES];
        char container_tag[LLAMA_HARD_MEMORY_MAX_TAG_CHARS];
    };

    struct llama_cognitive_hard_memory_result {
        int32_t command_id;
        int32_t tool_job_id;
        int32_t operation;
        struct llama_hard_memory_result result;
        struct llama_hard_memory_archive_trace archive_trace;
    };

    struct llama_telegram_relay_request {
        int32_t command_id;
        int32_t origin;
        int32_t tool_job_id;
        int32_t intent_kind;
        float urgency;
        bool command_ready;
        char dedupe_key[LLAMA_TELEGRAM_RELAY_DEDUPE_MAX_CHARS];
        char text[LLAMA_TELEGRAM_RELAY_TEXT_MAX_CHARS];
    };

    struct llama_telegram_relay_result {
        int32_t command_id;
        int32_t tool_job_id;
        int32_t intent_kind;
        bool delivered;
        int64_t delivered_at_ms;
        char dedupe_key[LLAMA_TELEGRAM_RELAY_DEDUPE_MAX_CHARS];
        char error_text[LLAMA_TELEGRAM_RELAY_ERROR_MAX_CHARS];
    };

    struct llama_telegram_ask_option_item {
        char label[LLAMA_TELEGRAM_ASK_OPTION_LABEL_MAX_CHARS];
    };

    struct llama_telegram_ask_options_request {
        int32_t command_id;
        int32_t origin;
        int32_t tool_job_id;
        float urgency;
        bool command_ready;
        int32_t option_count;
        char dedupe_key[LLAMA_TELEGRAM_RELAY_DEDUPE_MAX_CHARS];
        char chat_scope[LLAMA_TELEGRAM_CHAT_SCOPE_MAX_CHARS];
        char question[LLAMA_TELEGRAM_ASK_QUESTION_MAX_CHARS];
        struct llama_telegram_ask_option_item options[LLAMA_TELEGRAM_ASK_MAX_OPTIONS];
    };

    struct llama_telegram_ask_options_result {
        int32_t command_id;
        int32_t tool_job_id;
        bool delivered;
        int64_t delivered_at_ms;
        char dedupe_key[LLAMA_TELEGRAM_RELAY_DEDUPE_MAX_CHARS];
        char chat_scope[LLAMA_TELEGRAM_CHAT_SCOPE_MAX_CHARS];
        char error_text[LLAMA_TELEGRAM_RELAY_ERROR_MAX_CHARS];
    };

    struct llama_favorable_dimension_target {
        int32_t dimension_id;
        float current_value;
        float target_value;
        float tolerance;
        float weight;
        float divergence;
        float weighted_divergence;
        bool stable;
    };

    struct llama_favorable_state_profile {
        int32_t dimension_count;
        struct llama_favorable_dimension_target dimensions[LLAMA_FAVORABLE_MAX_DIMS];
        int32_t priority_count;
        int32_t priority_order[LLAMA_FAVORABLE_MAX_DIMS];
        float aggregate_divergence;
    };

    struct llama_counterfactual_candidate {
        int32_t family;
        int32_t risk_tier;
        int32_t subject_id;
        int32_t functional_target_kind;
        int32_t functional_family;
        int32_t process_entry_slot;
        int32_t proposal_family;
        int32_t replay_mode;
        int32_t snapshot_slot;
        float expected_improvement;
        float confidence;
        float fragility_penalty;
        float concentration_penalty;
        float robustness_score;
        float orthogonality;
        float realized_score;
        float signed_advantage_vs_current;
    };

    struct llama_user_simulation_trace {
        bool valid;
        bool used_user_personality_adapter;
        bool temporal_layers_ablated;
        int32_t source_family;
        int32_t prompt_token_count;
        int32_t reply_token_count;
        float simulation_confidence;
        float pre_simulation_divergence;
        float post_simulation_divergence;
        float signed_self_state_outcome;
        char candidate_message[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS];
        char simulated_user_reply[LLAMA_HARD_MEMORY_MAX_TEXT_CHARS];
    };

    struct llama_counterfactual_trace {
        int32_t candidate_count;
        struct llama_counterfactual_candidate candidates[LLAMA_COUNTERFACTUAL_MAX_CANDIDATES];
        int32_t winner_index;
        bool escalated;
        int32_t escalation_family;
        struct llama_user_simulation_trace simulated_user;
    };

    struct llama_remediation_plan {
        int32_t action;
        int32_t source_family;
        int32_t tool_kind;
        float expected_improvement;
        float confidence;
        float budget;
        int32_t tool_job_id;
        bool applied;
        float pre_divergence;
        float post_divergence;
    };

    struct llama_governance_trace {
        int32_t proposal_family;
        int32_t risk_tier;
        int32_t outcome;
        float evidence;
        float threshold;
        float dissatisfaction;
        float recent_user_valence;
        bool repair_rendered;
        int32_t repair_message_length;
        char repair_message[LLAMA_REPAIR_MESSAGE_MAX_CHARS];
    };

    struct llama_cognitive_tool_spec {
        int32_t tool_kind;
        uint32_t flags;
        int32_t latency_class;
        int32_t max_steps_reserved;
        char name[LLAMA_COGNITIVE_TOOL_NAME_MAX_CHARS];
        char description[LLAMA_COGNITIVE_TOOL_DESCRIPTION_MAX_CHARS];
        char capability_id[LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS];
        char owner_plugin_id[LLAMA_OPENCLAW_PLUGIN_ID_MAX_CHARS];
        char provenance_namespace[LLAMA_OPENCLAW_NAMESPACE_MAX_CHARS];
    };

    struct llama_cognitive_tool_proposal {
        bool valid;
        int32_t tool_kind;
        int32_t spec_index;
        uint32_t reason_mask;
        int32_t source_family;
        uint32_t safety_flags;
        int32_t expected_steps;
        float expected_observation_gain;
        int32_t job_id;
        char capability_id[LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS];
        char provenance_namespace[LLAMA_OPENCLAW_NAMESPACE_MAX_CHARS];
    };

    struct llama_cognitive_observation {
        bool valid;
        int32_t tool_kind;
        int32_t spec_index;
        int32_t job_id;
        int32_t status;
        float signal;
        float followup_affinity;
        char capability_id[LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS];
        char provenance_namespace[LLAMA_OPENCLAW_NAMESPACE_MAX_CHARS];
    };

    struct llama_cognitive_loop_state {
        int32_t phase;
        int32_t terminal_reason;
        int32_t max_steps;
        int32_t steps_taken;
        bool continuation_allowed;
        bool waiting_on_tool;
        int32_t tool_registry_count;
    };

    struct llama_cognitive_command {
        int32_t command_id;
        int32_t origin;
        int32_t kind;
        int32_t status;
        int32_t episode_id;
        int32_t tick_id;
        int32_t tool_kind;
        int32_t tool_spec_index;
        int32_t tool_job_id;
        uint32_t reason_mask;
        float priority;
        int32_t source_family;
        int32_t loop_phase;
        char capability_id[LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS];
    };

    struct llama_cognitive_plan_step {
        int32_t kind;
        int32_t status;
        int32_t tool_kind;
        int32_t tool_spec_index;
        int32_t source_family;
        uint32_t reason_mask;
        float priority;
        int32_t expected_steps;
        bool requires_tool_result;
        char capability_id[LLAMA_OPENCLAW_CAPABILITY_ID_MAX_CHARS];
        char provenance_namespace[LLAMA_OPENCLAW_NAMESPACE_MAX_CHARS];
    };

    struct llama_cognitive_plan_trace {
        bool valid;
        int32_t plan_id;
        int32_t origin;
        int32_t mode;
        int32_t status;
        int32_t revision_count;
        int32_t current_step_index;
        int32_t step_count;
        int32_t selected_family;
        int32_t terminal_reason;
        uint32_t reason_mask;
        float plan_score;
        float ambiguity;
        struct llama_cognitive_plan_step steps[LLAMA_COGNITIVE_MAX_PLAN_STEPS];
    };

    struct llama_cognitive_active_runner_status {
        int32_t episode_id;
        bool active;
        bool waiting_on_tool;
        bool completed;
        bool planning_active;
        int32_t steps_taken;
        int32_t max_steps;
        int32_t pending_command_id;
        int32_t pending_tool_spec_index;
        int32_t last_command_id;
        int32_t functional_microphase;
        int32_t plan_id;
        int32_t plan_mode;
        int32_t plan_status;
        int32_t plan_revision_count;
        int32_t current_plan_step;
    };

    struct llama_cognitive_dmn_runner_status {
        int32_t tick_id;
        bool active;
        bool waiting_on_tool;
        bool completed;
        bool planning_active;
        int32_t steps_taken;
        int32_t max_steps;
        int32_t pending_command_id;
        int32_t pending_tool_spec_index;
        int32_t last_command_id;
        int32_t functional_microphase;
        int32_t plan_id;
        int32_t plan_mode;
        int32_t plan_status;
        int32_t plan_revision_count;
        int32_t current_plan_step;
        int32_t self_model_revision_id;
        int32_t emotive_revision_id;
        int64_t context_revision;
        int32_t turn_id;
    };

    struct llama_active_loop_candidate {
        int32_t action;
        float score;
        float user_relevance;
        float latency_pressure;
        float tool_affinity;
        float inhibition;
        uint32_t reason_mask;
    };

    struct llama_active_loop_trace {
        int32_t episode_id;
        int32_t source_role;
        int32_t channel;
        uint32_t event_flags;
        uint64_t arrival_time_us;
        uint64_t completed_time_us;
        uint64_t shared_state_version;
        bool deferred_background;
        bool emit_allowed;
        bool emit_noted;
        bool tool_followup_expected;
        struct llama_cognitive_loop_state loop_state;
        struct llama_cognitive_tool_proposal tool_proposal;
        struct llama_cognitive_observation observation;
        struct llama_self_state_feature_vector prewrite_features;
        struct llama_self_state_feature_vector postwrite_features;
        int32_t candidate_count;
        struct llama_active_loop_candidate candidates[LLAMA_ACTIVE_LOOP_MAX_CANDIDATES];
        int32_t winner_action;
        float winner_score;
        int32_t runner_up_action;
        float runner_up_score;
        uint32_t reason_mask;
        struct llama_cognitive_plan_trace plan;
        struct llama_functional_activation_decision functional_activation;
        struct llama_self_model_revision self_model_revision;
        struct llama_emotive_moment_revision emotive_moment_revision;
        struct llama_shared_cognitive_context_window context_window;
        struct llama_authoritative_react_turn_state authoritative_turn;
    };

    struct llama_cognitive_host_state {
        uint64_t shared_state_version;
        int32_t active_episode_count;
        int32_t dmn_tick_count;
        int32_t background_deferred_count;
        int32_t pending_tool_followup_count;
        int32_t pending_dmn_emits;
        uint64_t last_foreground_time_us;
        uint64_t last_dmn_time_us;
    };

    struct llama_dmn_pressure_vector {
        float contradiction;
        float uncertainty;
        float reactivation;
        float goals;
        float tool_delta;
        float counterfactual;
        float continuation;
        float repair;
        float total;
    };

    struct llama_dmn_candidate {
        int32_t action;
        float score;
        float inhibition;
        float social_relevance;
        float continuation;
        float tool_affinity;
        uint32_t reason_mask;
    };

    struct llama_dmn_tick_trace {
        int32_t tick_id;
        bool admitted;
        bool deferred_for_foreground;
        struct llama_dmn_pressure_vector pressure;
        struct llama_dmn_self_model_revision self_model_revision;
        struct llama_self_model_revision canonical_self_model_revision;
        struct llama_emotive_moment_revision emotive_moment_revision;
        struct llama_shared_cognitive_context_window context_window;
        struct llama_authoritative_react_turn_state authoritative_turn;
        int32_t reactivation_count;
        struct llama_self_reactivation_info reactivation_targets[LLAMA_DMN_MAX_REACTIVATION_TARGETS];
        uint32_t seed_source_mask;
        float seed_dims[LLAMA_DMN_SEED_DIMS];
        int32_t candidate_count;
        struct llama_dmn_candidate candidates[LLAMA_DMN_MAX_CANDIDATES];
        int32_t winner_action;
        float winner_score;
        int32_t runner_up_action;
        float runner_up_score;
        int32_t burst_count;
        uint32_t maintenance_mask;
        int32_t tool_kind;
        int32_t tool_spec_index;
        int32_t tool_job_id;
        float favorable_divergence;
        char reasoning_text[LLAMA_DMN_REASONING_MAX_CHARS];
        struct llama_cognitive_plan_trace plan;
        struct llama_functional_activation_decision functional_activation;
        struct llama_cognitive_loop_state loop_state;
        struct llama_cognitive_tool_proposal tool_proposal;
        struct llama_cognitive_observation observation;
        struct llama_user_simulation_trace simulated_user;
    };

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLAMA_API struct llama_model_params          llama_model_default_params(void);
    LLAMA_API struct llama_context_params        llama_context_default_params(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);
    LLAMA_API struct llama_active_lora_params    llama_active_lora_default_params(void);
    LLAMA_API struct llama_past_lora_params      llama_past_lora_default_params(void);
    LLAMA_API struct llama_process_functional_params llama_process_functional_default_params(void);
    LLAMA_API struct llama_self_state_params     llama_self_state_default_params(void);
    LLAMA_API struct llama_self_updater_program  llama_self_state_default_updater_program(void);
    LLAMA_API struct llama_hard_memory_config    llama_hard_memory_default_config(void);
    LLAMA_API struct llama_hard_memory_primitive llama_hard_memory_default_primitive(void);
    LLAMA_API struct llama_bash_tool_config      llama_bash_tool_default_config(void);

    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free(void);

    //optional:
    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    LLAMA_API void llama_attach_threadpool(
            struct llama_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);

    typedef void (*llama_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    // Create a new model from GGUF metadata as well as a function to set the tensor data
    //   - tensors are created as GGML_TYPE_F32 by default,
    //     override by adding a tensor with the same name but a different name to the context
    LLAMA_API struct llama_model * llama_model_init_from_user(
                    struct gguf_context * metadata,
          llama_model_set_tensor_data_t   set_tensor_data,    // function to initialize tensor data with
                                   void * set_tensor_data_ud, // userdata for function
              struct llama_model_params   params);

    DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
              struct llama_model_params   params),
            "use llama_model_load_from_file instead");

    // Load a model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use llama_model_load_from_splits
    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
              struct llama_model_params   params);

    // Load a model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    LLAMA_API struct llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct llama_model_params    params);

    LLAMA_API void llama_model_save_to_file(
            const struct llama_model * model,
                        const char * path_model);

    DEPRECATED(LLAMA_API void llama_free_model(struct llama_model * model),
            "use llama_model_free instead");

    LLAMA_API void llama_model_free(struct llama_model * model);

    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    DEPRECATED(LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params),
            "use llama_init_from_model instead");

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context * ctx);

    enum llama_params_fit_status {
        LLAMA_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
        LLAMA_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
        LLAMA_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
    };

    // fits mparams and cparams to free device memory (assumes system memory is unlimited)
    //   - returns true if the parameters could be successfully modified to fit device memory
    //   - this function is NOT thread safe because it modifies the global llama logger state
    //   - only parameters that have the same value as in llama_default_model_params are modified
    //     with the exception of the context size which is modified if and only if equal to 0
    LLAMA_API enum llama_params_fit_status llama_params_fit(
                                   const char   * path_model,
                    struct llama_model_params   * mparams,
                    struct llama_context_params * cparams,
                                          float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
        struct llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                                         size_t * margins,               // margins of memory to leave per device in bytes
                                       uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
                            enum ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);
    LLAMA_API size_t llama_max_parallel_sequences(void);
    LLAMA_API size_t llama_max_tensor_buft_overrides(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);
    LLAMA_API bool llama_supports_rpc        (void);

    // NOTE: After creating a llama_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via llama_context_params may differ from the actual values used by the context
    //       ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ctx_seq  (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);

    DEPRECATED(LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model), "use llama_model_n_ctx_train instead");
    DEPRECATED(LLAMA_API int32_t llama_n_embd     (const struct llama_model * model), "use llama_model_n_embd instead");
    DEPRECATED(LLAMA_API int32_t llama_n_layer    (const struct llama_model * model), "use llama_model_n_layer instead");
    DEPRECATED(LLAMA_API int32_t llama_n_head     (const struct llama_model * model), "use llama_model_n_head instead");

    DEPRECATED(LLAMA_API int32_t llama_n_vocab    (const struct llama_vocab * vocab), "use llama_vocab_n_tokens instead");

    LLAMA_API const struct llama_model * llama_get_model   (const struct llama_context * ctx);
    LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
    LLAMA_API  enum llama_pooling_type   llama_pooling_type(const struct llama_context * ctx); // TODO: rename to llama_get_pooling_type

    LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    LLAMA_API enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);

    LLAMA_API int32_t llama_model_n_ctx_train(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_inp (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_out (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer    (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head_kv  (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_swa      (const struct llama_model * model);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    LLAMA_API uint32_t llama_model_n_cls_out(const struct llama_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    LLAMA_API const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);

    LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);

    LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    LLAMA_API const char * llama_model_meta_key_str(enum llama_model_meta_key key);

    // Get metadata key name by index
    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // Returns true if the model contains an encoder that requires llama_encode() call
    LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);

    // Returns true if the model contains a decoder that requires llama_decode() call
    LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    LLAMA_API bool llama_model_is_hybrid(const struct llama_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    LLAMA_API bool llama_model_is_diffusion(const struct llama_model * model);

    // Returns 0 on success
    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    // All adapters must be loaded before context creation
    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
            struct llama_model * model,
            const char * path_lora);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_adapter_meta_val_str(const struct llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_adapter_meta_count(const struct llama_adapter_lora * adapter);

    // Get metadata key name by index
    LLAMA_API int32_t llama_adapter_meta_key_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_adapter_meta_val_str_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // NOTE: loaded adapters will be free when the associated model is deleted
    LLAMA_API DEPRECATED(void llama_adapter_lora_free(struct llama_adapter_lora * adapter),
            "adapters are now freed together with the associated model");

    // Get the invocation tokens if the current lora is an alora
    LLAMA_API uint64_t            llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter);
    LLAMA_API const llama_token * llama_adapter_get_alora_invocation_tokens  (const struct llama_adapter_lora * adapter);

    // The following functions operate on a llama_context, hence the naming: llama_verb_...

    // Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
    LLAMA_API int32_t llama_set_adapters_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Apply a loaded control vector to a llama_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llama_control_vector_load in common to load a control vector.
    LLAMA_API int32_t llama_set_adapter_cvec(
            struct llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    // Initialize a runtime-generated Active LoRA memory stage on the context.
    LLAMA_API int32_t llama_active_lora_init(
            struct llama_context * ctx,
            struct llama_active_lora_params params);

    // Ingest an evicted token span into the Active LoRA memory stage.
    LLAMA_API int32_t llama_active_lora_ingest(
            struct llama_context * ctx,
            const llama_token * tokens,
            size_t n_tokens);
    LLAMA_API int32_t llama_active_lora_ingest_event(
            struct llama_context * ctx,
            const struct llama_self_state_event * event,
            const struct llama_self_state_feature_vector * features);

    // Inspect Active LoRA state.
    LLAMA_API int32_t llama_active_lora_get_stats(
            const struct llama_context * ctx,
            struct llama_active_lora_stats * out_stats);
    LLAMA_API int32_t llama_user_personality_lora_get_stats(
            const struct llama_context * ctx,
            struct llama_user_personality_lora_stats * out_stats);

    // Initialize the frozen temporal past-LoRA stack on the context.
    LLAMA_API int32_t llama_past_lora_init(
            struct llama_context * ctx,
            struct llama_past_lora_params params);

    // Advance decay and condensation jobs for the frozen past-LoRA stack.
    LLAMA_API int32_t llama_past_lora_tick(
            struct llama_context * ctx,
            uint64_t now_us);

    // Inspect frozen past-LoRA stack state.
    LLAMA_API int32_t llama_past_lora_get_stats(
            const struct llama_context * ctx,
            struct llama_past_lora_stats * out_stats);

    // Inspect the effective live serving LoRA stack after request adapters and runtime memory layers
    // have been composed for decode.
    LLAMA_API int32_t llama_serving_lora_stack_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_serving_lora_stack_layer(
            const struct llama_context * ctx,
            int32_t i,
            struct llama_serving_lora_layer_info * out_info);
    LLAMA_API int32_t llama_functional_lora_family_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_functional_lora_family_config_get(
            const struct llama_context * ctx,
            int32_t family,
            struct llama_functional_lora_family_config * out_config);
    LLAMA_API int32_t llama_functional_lora_family_state_get(
            const struct llama_context * ctx,
            int32_t family,
            struct llama_functional_lora_family_state * out_state);
    LLAMA_API int32_t llama_functional_lora_get_last_trace(
            const struct llama_context * ctx,
            struct llama_functional_lora_trace * out_trace);
    LLAMA_API int32_t llama_functional_lora_get_last_update(
            const struct llama_context * ctx,
            int32_t family,
            struct llama_functional_lora_update_info * out_update);
    LLAMA_API int32_t llama_functional_lora_snapshot_archive_get(
            const struct llama_context * ctx,
            int32_t family,
            struct llama_functional_lora_snapshot_archive * out_archive);
    LLAMA_API int32_t llama_functional_lora_snapshot_info_get(
            const struct llama_context * ctx,
            int32_t family,
            int32_t slot,
            struct llama_functional_lora_snapshot_info * out_info);
    LLAMA_API int32_t llama_functional_lora_get_last_snapshot_maintenance(
            const struct llama_context * ctx,
            struct llama_functional_snapshot_maintenance_trace * out_trace);
    LLAMA_API int32_t llama_functional_lora_set_ablation(
            struct llama_context * ctx,
            struct llama_functional_lora_ablation_config config);
    LLAMA_API int32_t llama_functional_lora_get_ablation(
            const struct llama_context * ctx,
            struct llama_functional_lora_ablation_config * out_config);
    LLAMA_API int32_t llama_functional_lora_replay_override_begin(
            struct llama_context * ctx,
            struct llama_functional_lora_replay_override config);
    LLAMA_API int32_t llama_functional_lora_replay_override_end(
            struct llama_context * ctx,
            int32_t family);
    LLAMA_API int32_t llama_functional_lora_get_last_differential_update(
            const struct llama_context * ctx,
            int32_t family,
            struct llama_functional_lora_differential_update * out_update);
    LLAMA_API size_t llama_functional_lora_snapshot_blob_size(
            const struct llama_context * ctx,
            int32_t family,
            int32_t slot);
    LLAMA_API int32_t llama_functional_lora_snapshot_blob_export(
            const struct llama_context * ctx,
            int32_t family,
            int32_t slot,
            void * dst,
            size_t size);
    LLAMA_API int32_t llama_functional_lora_snapshot_blob_import(
            struct llama_context * ctx,
            int32_t family,
            int32_t slot,
            struct llama_functional_lora_snapshot_info info,
            const void * src,
            size_t size);
    LLAMA_API int32_t llama_functional_lora_snapshot_maintain(
            struct llama_context * ctx,
            uint64_t now_us);
    LLAMA_API int32_t llama_process_functional_get_params(
            const struct llama_context * ctx,
            struct llama_process_functional_params * out_params);
    LLAMA_API int32_t llama_process_functional_set_params(
            struct llama_context * ctx,
            struct llama_process_functional_params params);
    LLAMA_API int32_t llama_process_functional_entry_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_process_functional_entry_get(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_process_functional_entry_info * out_info);
    LLAMA_API int32_t llama_process_functional_ledger_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_process_functional_ledger_get(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_process_functional_ledger_info * out_info);
    LLAMA_API int32_t llama_process_functional_get_last_trace(
            const struct llama_context * ctx,
            struct llama_process_functional_trace * out_trace);
    LLAMA_API int32_t llama_process_functional_get_current_signature(
            const struct llama_context * ctx,
            struct llama_process_functional_signature * out_signature);
    LLAMA_API int32_t llama_process_functional_snapshot_archive_get(
            const struct llama_context * ctx,
            int32_t entry_slot,
            struct llama_functional_lora_snapshot_archive * out_archive);
    LLAMA_API int32_t llama_process_functional_snapshot_info_get(
            const struct llama_context * ctx,
            int32_t entry_slot,
            int32_t snapshot_slot,
            struct llama_functional_lora_snapshot_info * out_info);
    LLAMA_API int32_t llama_process_functional_get_last_snapshot_maintenance(
            const struct llama_context * ctx,
            struct llama_functional_snapshot_maintenance_trace * out_trace);
    LLAMA_API int32_t llama_process_functional_replay_override_begin(
            struct llama_context * ctx,
            int32_t entry_slot,
            struct llama_functional_lora_replay_override config);
    LLAMA_API int32_t llama_process_functional_replay_override_end(
            struct llama_context * ctx,
            int32_t entry_slot);
    LLAMA_API int32_t llama_process_functional_get_last_differential_update(
            const struct llama_context * ctx,
            int32_t entry_slot,
            struct llama_functional_lora_differential_update * out_update);
    LLAMA_API int32_t llama_process_functional_apply_differential_update(
            struct llama_context * ctx,
            int32_t entry_slot,
            int32_t proposal_family,
            int32_t replay_mode,
            int32_t snapshot_slot,
            float signed_score_delta,
            float magnitude,
            float robustness_score);
    LLAMA_API size_t llama_process_functional_entry_blob_size(
            const struct llama_context * ctx,
            int32_t index);
    LLAMA_API int32_t llama_process_functional_entry_blob_export(
            const struct llama_context * ctx,
            int32_t index,
            void * dst,
            size_t size);
    LLAMA_API int32_t llama_process_functional_entry_blob_import(
            struct llama_context * ctx,
            int32_t index,
            const struct llama_process_functional_entry_info * info,
            const void * src,
            size_t size);
    LLAMA_API size_t llama_process_functional_snapshot_blob_size(
            const struct llama_context * ctx,
            int32_t entry_slot,
            int32_t snapshot_slot);
    LLAMA_API int32_t llama_process_functional_snapshot_blob_export(
            const struct llama_context * ctx,
            int32_t entry_slot,
            int32_t snapshot_slot,
            void * dst,
            size_t size);
    LLAMA_API int32_t llama_process_functional_snapshot_blob_import(
            struct llama_context * ctx,
            int32_t entry_slot,
            int32_t snapshot_slot,
            struct llama_functional_lora_snapshot_info info,
            const void * src,
            size_t size);
    LLAMA_API int32_t llama_process_functional_snapshot_maintain(
            struct llama_context * ctx,
            uint64_t now_us);
    LLAMA_API int32_t llama_active_temporal_encoding_bias_get(
            const struct llama_context * ctx,
            struct llama_active_temporal_encoding_bias * out_bias);
    LLAMA_API int32_t llama_temporal_self_improvement_get_last(
            const struct llama_context * ctx,
            struct llama_temporal_self_improvement_trace * out_trace);

    // Refresh the self-state time surface from the local system clock.
    LLAMA_API int32_t llama_self_state_refresh_time(struct llama_context * ctx);

    // Apply a deterministic time point to the self-state for replay and testing.
    LLAMA_API int32_t llama_self_state_set_time(
            struct llama_context * ctx,
            struct llama_self_state_time_point time_point);

    // Inspect the expanded self-state datetime surface.
    LLAMA_API int32_t llama_self_state_get_datetime(
            const struct llama_context * ctx,
            struct llama_self_state_datetime * out_info);

    // Inspect the predefined self-state register bank.
    LLAMA_API int32_t llama_self_state_register_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_get_register(
            const struct llama_context * ctx,
            int32_t register_id,
            struct llama_self_register_info * out_info);
    LLAMA_API const char * llama_self_state_register_name(int32_t register_id);
    LLAMA_API int32_t llama_self_state_configure(
            struct llama_context * ctx,
            struct llama_self_state_params params);

    // Update explicit self-state control surfaces and event anchors.
    LLAMA_API int32_t llama_self_state_set_channel_state(
            struct llama_context * ctx,
            int32_t channel_state);
    LLAMA_API int32_t llama_self_state_note_user_event(struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_note_tool_event(struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_note_emit_event(struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_set_identity(
            struct llama_context * ctx,
            const llama_token * tokens,
            size_t n_tokens);
    LLAMA_API int32_t llama_self_state_upsert_goal(
            struct llama_context * ctx,
            int32_t goal_id,
            const llama_token * tokens,
            size_t n_tokens,
            float priority);
    LLAMA_API int32_t llama_self_state_upsert_commitment(
            struct llama_context * ctx,
            int32_t commitment_id,
            const llama_token * tokens,
            size_t n_tokens,
            float priority,
            bool unresolved);
    LLAMA_API int32_t llama_self_state_goal_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_commitment_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_working_memory_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_upsert_memory_handle(
            struct llama_context * ctx,
            int32_t handle_id,
            int32_t kind,
            const llama_token * tokens,
            size_t n_tokens,
            float priority);
    LLAMA_API int32_t llama_self_state_memory_handle_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_reactivation_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_get_reactivation(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_self_reactivation_info * out_info);
    LLAMA_API int32_t llama_self_state_upsert_tool_job(
            struct llama_context * ctx,
            int32_t job_id,
            int32_t status,
            float importance);
    LLAMA_API int32_t llama_self_state_get_tool_state(
            const struct llama_context * ctx,
            struct llama_self_tool_state_info * out_info);
    LLAMA_API int32_t llama_self_state_get_social_state(
            const struct llama_context * ctx,
            struct llama_self_social_state_info * out_info);
    LLAMA_API int32_t llama_self_state_get_model_state(
            const struct llama_context * ctx,
            struct llama_self_model_state_info * out_info);

    LLAMA_API int32_t llama_self_state_get_self_model_revision(
            const struct llama_context * ctx,
            struct llama_self_model_revision * out_info);

    LLAMA_API int32_t llama_self_state_get_emotive_moment_revision(
            const struct llama_context * ctx,
            struct llama_emotive_moment_revision * out_info);
    LLAMA_API int32_t llama_self_state_get_last_disturbance(
            const struct llama_context * ctx,
            struct llama_self_disturbance_state_info * out_info);
    LLAMA_API struct llama_self_model_extension_update llama_self_model_extension_default_update(void);
    LLAMA_API int32_t llama_self_state_model_extension_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_get_model_extension(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_self_model_extension_info * out_info);
    LLAMA_API int32_t llama_self_state_upsert_model_extension(
            struct llama_context * ctx,
            struct llama_self_model_extension_update update);
    LLAMA_API int32_t llama_self_state_remove_model_extension(
            struct llama_context * ctx,
            const char * key);
    LLAMA_API int32_t llama_self_state_trace_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_trace_token_count(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_trace_get_item(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_self_trace_item_info * out_info);

    LLAMA_API int32_t llama_shared_cognitive_context_count(const struct llama_context * ctx);

    LLAMA_API int32_t llama_shared_cognitive_context_get_item(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_shared_cognitive_context_item * out_info);

    LLAMA_API int32_t llama_shared_cognitive_context_get_window(
            const struct llama_context * ctx,
            struct llama_shared_cognitive_context_window * out_info);
    LLAMA_API int32_t llama_self_state_clear_trace(struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_replay_trace(struct llama_context * ctx, int32_t upto_count);
    LLAMA_API int32_t llama_self_state_replay_trace_on_channel(
            struct llama_context * ctx,
            int32_t upto_count,
            int32_t replay_channel);
    LLAMA_API int32_t llama_self_state_set_updater_program(
            struct llama_context * ctx,
            struct llama_self_updater_program program);
    LLAMA_API int32_t llama_self_state_get_updater_program(
            const struct llama_context * ctx,
            struct llama_self_updater_program * out_program);
    LLAMA_API size_t llama_self_state_trace_export_size(const struct llama_context * ctx);
    LLAMA_API int32_t llama_self_state_trace_export(
            const struct llama_context * ctx,
            void * dst,
            size_t size);
    LLAMA_API int32_t llama_self_state_trace_import(
            struct llama_context * ctx,
            const void * src,
            size_t size,
            bool replace_existing);
    LLAMA_API int32_t llama_self_state_evaluate_counterfactual(
            const struct llama_context * ctx,
            struct llama_self_updater_program program,
            int32_t upto_count,
            struct llama_self_counterfactual_result * out_result);
    LLAMA_API int32_t llama_self_state_evaluate_counterfactual_on_channel(
            const struct llama_context * ctx,
            struct llama_self_updater_program program,
            int32_t upto_count,
            int32_t replay_channel,
            struct llama_self_counterfactual_result * out_result);

    // Build and apply explicit feature vectors for register recomputation.
    LLAMA_API int32_t llama_self_state_build_prewrite_features(
            const struct llama_context * ctx,
            const struct llama_self_state_event * event,
            struct llama_self_state_feature_vector * out_features);
    LLAMA_API int32_t llama_self_state_apply_prewrite(
            struct llama_context * ctx,
            const struct llama_self_state_event * event,
            const struct llama_self_state_feature_vector * features);
    LLAMA_API int32_t llama_self_state_build_postwrite_features(
            const struct llama_context * ctx,
            const struct llama_self_state_event * event,
            struct llama_self_state_feature_vector * out_features);
    LLAMA_API int32_t llama_self_state_apply_postwrite(
            struct llama_context * ctx,
            const struct llama_self_state_event * event,
            const struct llama_self_state_feature_vector * features);
    LLAMA_API int32_t llama_self_state_note_validated_progress(
            struct llama_context * ctx,
            float signed_progress,
            float efficiency_advantage);
    LLAMA_API int32_t llama_hard_memory_configure(
            struct llama_context * ctx,
            struct llama_hard_memory_config config);
    LLAMA_API int32_t llama_hard_memory_get_config(
            const struct llama_context * ctx,
            struct llama_hard_memory_config * out_config);
    LLAMA_API int32_t llama_hard_memory_query(
            struct llama_context * ctx,
            const struct llama_hard_memory_query_request * query,
            struct llama_hard_memory_result * out_result);
    LLAMA_API int32_t llama_hard_memory_archive_primitives(
            struct llama_context * ctx,
            const struct llama_hard_memory_primitive * primitives,
            int32_t primitive_count);
    LLAMA_API int32_t llama_hard_memory_get_last_result(
            const struct llama_context * ctx,
            struct llama_hard_memory_result * out_result);
    LLAMA_API int32_t llama_hard_memory_get_last_archive_trace(
            const struct llama_context * ctx,
            struct llama_hard_memory_archive_trace * out_trace);
    LLAMA_API int32_t llama_bash_tool_configure(
            struct llama_context * ctx,
            const struct llama_bash_tool_config * config);
    LLAMA_API int32_t llama_bash_tool_get_config(
            const struct llama_context * ctx,
            struct llama_bash_tool_config * out_config);
    LLAMA_API int32_t llama_bash_tool_get_last_result(
            const struct llama_context * ctx,
            struct llama_bash_tool_result * out_result);
    LLAMA_API struct llama_codex_tool_config llama_codex_tool_default_config(void);
    LLAMA_API int32_t llama_codex_tool_configure(
            struct llama_context * ctx,
            const struct llama_codex_tool_config * config);
    LLAMA_API int32_t llama_codex_tool_get_config(
            const struct llama_context * ctx,
            struct llama_codex_tool_config * out_config);
    LLAMA_API int32_t llama_codex_tool_get_last_result(
            const struct llama_context * ctx,
            struct llama_codex_tool_result * out_result);

    LLAMA_API int32_t llama_cognitive_tool_spec_count(
            const struct llama_context * ctx);
    LLAMA_API int32_t llama_cognitive_tool_spec_get(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_cognitive_tool_spec * out_spec);
    LLAMA_API int32_t llama_cognitive_tool_spec_set(
            struct llama_context * ctx,
            const struct llama_cognitive_tool_spec * specs,
            int32_t count);
    LLAMA_API int32_t llama_cognitive_command_count(
            const struct llama_context * ctx);
    LLAMA_API int32_t llama_cognitive_command_get(
            const struct llama_context * ctx,
            int32_t index,
            struct llama_cognitive_command * out_command);
    LLAMA_API int32_t llama_cognitive_command_ack(
            struct llama_context * ctx,
            int32_t command_id);
    LLAMA_API int32_t llama_cognitive_command_complete(
            struct llama_context * ctx,
            int32_t command_id,
            bool cancelled);
    LLAMA_API int32_t llama_cognitive_command_begin_external_wait(
            struct llama_context * ctx,
            int32_t command_id);
    LLAMA_API int32_t llama_cognitive_command_rebind_tool(
            struct llama_context * ctx,
            int32_t command_id,
            int32_t tool_spec_index);
    LLAMA_API int32_t llama_cognitive_authoritative_react_set_enabled(
            struct llama_context * ctx,
            bool enabled);
    LLAMA_API int32_t llama_cognitive_active_authoritative_prepare(
            struct llama_context * ctx,
            const struct llama_self_state_event * event,
            struct llama_active_loop_trace * out_trace);
    LLAMA_API int32_t llama_cognitive_active_authoritative_begin_tool(
            struct llama_context * ctx,
            int32_t episode_id,
            uint32_t reason_mask,
            float priority,
            int32_t * out_command_id,
            int32_t * out_tool_job_id);
    LLAMA_API int32_t llama_cognitive_active_authoritative_finish(
            struct llama_context * ctx,
            int32_t episode_id,
            int32_t terminal_reason);
    LLAMA_API int32_t llama_cognitive_dmn_authoritative_begin_tool(
            struct llama_context * ctx,
            int32_t tick_id,
            uint32_t reason_mask,
            float priority,
            int32_t * out_command_id,
            int32_t * out_tool_job_id);
    LLAMA_API int32_t llama_cognitive_dmn_authoritative_finish(
            struct llama_context * ctx,
            int32_t tick_id,
            int32_t terminal_reason);
    LLAMA_API int32_t llama_cognitive_active_tool_emission_note(
            struct llama_context * ctx,
            int32_t command_id,
            const llama_token * tokens,
            size_t n_tokens);
    LLAMA_API int32_t llama_cognitive_active_planner_reasoning_note(
            struct llama_context * ctx,
            int32_t episode_id,
            const llama_token * tokens,
            size_t n_tokens);
    LLAMA_API int32_t llama_cognitive_bash_tool_get_request(
            const struct llama_context * ctx,
            int32_t command_id,
            struct llama_bash_tool_request * out_request);
    LLAMA_API int32_t llama_cognitive_bash_tool_set_request(
            struct llama_context * ctx,
            const struct llama_bash_tool_request * request);
    LLAMA_API int32_t llama_cognitive_bash_tool_submit_result(
            struct llama_context * ctx,
            const struct llama_bash_tool_result * result,
            struct llama_active_loop_trace * out_active_trace);
    LLAMA_API int32_t llama_cognitive_codex_tool_get_request(
            const struct llama_context * ctx,
            int32_t command_id,
            struct llama_codex_tool_request * out_request);
    LLAMA_API int32_t llama_cognitive_codex_tool_set_request(
            struct llama_context * ctx,
            const struct llama_codex_tool_request * request);
    LLAMA_API int32_t llama_cognitive_codex_tool_submit_result(
            struct llama_context * ctx,
            const struct llama_codex_tool_result * result,
            struct llama_active_loop_trace * out_active_trace);
    LLAMA_API int32_t llama_cognitive_hard_memory_get_request(
            const struct llama_context * ctx,
            int32_t command_id,
            struct llama_cognitive_hard_memory_request * out_request);
    LLAMA_API int32_t llama_cognitive_hard_memory_set_request(
            struct llama_context * ctx,
            const struct llama_cognitive_hard_memory_request * request);
    LLAMA_API int32_t llama_cognitive_hard_memory_submit_result(
            struct llama_context * ctx,
            const struct llama_cognitive_hard_memory_result * result,
            struct llama_active_loop_trace * out_active_trace);
    LLAMA_API int32_t llama_cognitive_telegram_relay_get_request(
            const struct llama_context * ctx,
            int32_t command_id,
            struct llama_telegram_relay_request * out_request);
    LLAMA_API int32_t llama_cognitive_telegram_relay_set_request(
            struct llama_context * ctx,
            const struct llama_telegram_relay_request * request);
    LLAMA_API int32_t llama_cognitive_telegram_relay_submit_result(
            struct llama_context * ctx,
            const struct llama_telegram_relay_result * result,
            struct llama_active_loop_trace * out_active_trace);
    LLAMA_API int32_t llama_cognitive_telegram_ask_options_get_request(
            const struct llama_context * ctx,
            int32_t command_id,
            struct llama_telegram_ask_options_request * out_request);
    LLAMA_API int32_t llama_cognitive_telegram_ask_options_set_request(
            struct llama_context * ctx,
            const struct llama_telegram_ask_options_request * request);
    LLAMA_API int32_t llama_cognitive_telegram_ask_options_submit_result(
            struct llama_context * ctx,
            const struct llama_telegram_ask_options_result * result,
            struct llama_active_loop_trace * out_active_trace);
    LLAMA_API int32_t llama_cognitive_active_runner_get(
            const struct llama_context * ctx,
            struct llama_cognitive_active_runner_status * out_status);
    LLAMA_API int32_t llama_cognitive_dmn_runner_get(
            const struct llama_context * ctx,
            struct llama_cognitive_dmn_runner_status * out_status);

    // Run the active engagement loop for one user or tool event.
    LLAMA_API int32_t llama_active_loop_process(
            struct llama_context * ctx,
            const struct llama_self_state_event * event,
            struct llama_active_loop_trace * out_trace);
    LLAMA_API int32_t llama_active_loop_note_emit(
            struct llama_context * ctx,
            int32_t episode_id,
            size_t emitted_text_bytes);
    LLAMA_API int32_t llama_active_loop_get_last_trace(
            const struct llama_context * ctx,
            struct llama_active_loop_trace * out_trace);

    // Run or defer the pressure-driven DMN runtime.
    LLAMA_API int32_t llama_dmn_tick(
            struct llama_context * ctx,
            uint64_t now_us,
            struct llama_dmn_tick_trace * out_trace);
    LLAMA_API int32_t llama_dmn_defer(
            struct llama_context * ctx,
            uint64_t now_us,
            struct llama_dmn_tick_trace * out_trace);
    LLAMA_API int32_t llama_dmn_get_last_trace(
            const struct llama_context * ctx,
            struct llama_dmn_tick_trace * out_trace);
    LLAMA_API int32_t llama_cognitive_get_host_state(
            const struct llama_context * ctx,
            struct llama_cognitive_host_state * out_state);
    LLAMA_API int32_t llama_favorable_state_get(
            const struct llama_context * ctx,
            struct llama_favorable_state_profile * out_profile);
    LLAMA_API int32_t llama_counterfactual_get_last_trace(
            const struct llama_context * ctx,
            struct llama_counterfactual_trace * out_trace);
    LLAMA_API int32_t llama_remediation_get_last_plan(
            const struct llama_context * ctx,
            struct llama_remediation_plan * out_plan);
    LLAMA_API int32_t llama_governance_get_last_trace(
            const struct llama_context * ctx,
            struct llama_governance_trace * out_trace);

    //
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    LLAMA_API void llama_memory_clear(
            llama_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_memory_seq_rm(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_cp(
            llama_memory_t mem,
              llama_seq_id seq_id_src,
              llama_seq_id seq_id_dst,
                 llama_pos p0,
                 llama_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_memory_seq_keep(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_add(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                 llama_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_div(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_min(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_max(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Check if the memory supports shifting
    LLAMA_API bool llama_memory_can_shift(llama_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
        "use llama_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    // Save/load session file
    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id);

    // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id);

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

// for backwards-compat
#define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

    typedef uint32_t llama_state_seq_flags;

    LLAMA_API size_t llama_state_seq_get_size_ext(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_get_data_ext(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_set_data_ext(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by llama_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llama_batch_free()
    // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    // The rest of the llama_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llama_batch_init()
    LLAMA_API void llama_batch_free(struct llama_batch batch);

    // Process a batch of tokens.
    // In contrast to llama_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    LLAMA_API int32_t llama_encode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with llama_get_embeddings()
    LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during llama_decode() to load and cache their weights.
    LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);

    // Set abort callback
    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of llama_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indices can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of llama_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    // when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the llama_context was created with at least one llama_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // Returns LLAMA_TOKEN_NULL if no token was sampled.
    LLAMA_API llama_token llama_get_sampled_token_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled probabilities for the ith token
    // The index matches llama_get_sampled_token_ith().
    // Returns NULL if no probabilities were generated.
    LLAMA_API float *  llama_get_sampled_probs_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    LLAMA_API float *  llama_get_sampled_logits_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    LLAMA_API llama_token * llama_get_sampled_candidates_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t      llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);

    //
    // Vocab
    //

    LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    // Identify if Token Id is a control token or a render-able token
    LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
    LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
    LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab); // end-of-turn
    LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab); // sentence separator
    LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab); // next-line
    LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab); // padding
    LLAMA_API llama_token llama_vocab_mask(const struct llama_vocab * vocab); // mask

    LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);

    LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);

    DEPRECATED(LLAMA_API const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_text instead");
    DEPRECATED(LLAMA_API float llama_token_get_score(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_score instead");
    DEPRECATED(LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_attr instead");
    DEPRECATED(LLAMA_API bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_eog instead");
    DEPRECATED(LLAMA_API bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_control instead");
    DEPRECATED(LLAMA_API llama_token llama_token_bos(const struct llama_vocab * vocab), "use llama_vocab_bos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eos(const struct llama_vocab * vocab), "use llama_vocab_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eot(const struct llama_vocab * vocab), "use llama_vocab_eot instead");
    DEPRECATED(LLAMA_API llama_token llama_token_cls(const struct llama_vocab * vocab), "use llama_vocab_cls instead");
    DEPRECATED(LLAMA_API llama_token llama_token_sep(const struct llama_vocab * vocab), "use llama_vocab_sep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_nl (const struct llama_vocab * vocab), "use llama_vocab_nl instead");
    DEPRECATED(LLAMA_API llama_token llama_token_pad(const struct llama_vocab * vocab), "use llama_vocab_pad instead");
    DEPRECATED(LLAMA_API bool llama_add_bos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_bos instead");
    DEPRECATED(LLAMA_API bool llama_add_eos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pre(const struct llama_vocab * vocab), "use llama_vocab_fim_pre instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_suf(const struct llama_vocab * vocab), "use llama_vocab_fim_suf instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_mid(const struct llama_vocab * vocab), "use llama_vocab_fim_mid instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pad(const struct llama_vocab * vocab), "use llama_vocab_fim_pad instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_rep(const struct llama_vocab * vocab), "use llama_vocab_fim_rep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_sep(const struct llama_vocab * vocab), "use llama_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(LLAMA_API llama_token llama_vocab_cls(const struct llama_vocab * vocab), // classification
            "use llama_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLAMA_API int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_detokenize(
        const struct llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //

    typedef void * llama_sampler_context_t;

    struct llama_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom llama_sampler
    struct llama_sampler_i {
        const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler
        // note: call once per sampler
        bool (*backend_init)(struct llama_sampler * smpl, ggml_backend_buffer_type_t buft);

        // call after .backend_apply()
        void (*backend_accept)(
                struct llama_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct llama_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct llama_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct llama_sampler * smpl);
    };

    struct llama_sampler {
        struct llama_sampler_i * iface;

        llama_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with llama_context_params.samplers when possible
    LLAMA_API bool llama_set_sampler(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * smpl);

    // mirror of llama_sampler_i:
    LLAMA_API struct llama_sampler * llama_sampler_init  (      struct llama_sampler_i * iface, llama_sampler_context_t ctx);
    LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
    LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
    LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
    // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);

    // llama_sampler_chain
    // a type of llama_sampler that can chain multiple samplers one after another

    LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a llama_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    LLAMA_API struct llama_sampler * llama_sampler_chain_get(      struct llama_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    LLAMA_API int                    llama_sampler_chain_n  (const struct llama_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);

    // available samplers:

    LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);

    /// seed == LLAMA_DEFAULT_SEED to use a random seed.
    LLAMA_API struct llama_sampler * llama_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Initializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llama_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use llama_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
            const struct llama_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: https://github.com/ggml-org/llama.cpp/pull/17927
    ///
    LLAMA_API struct llama_sampler * llama_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = llama_get_logits_ith(ctx, idx);
    //    llama_token_data_array cur_p = { ... init from logits ... };
    //    llama_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    llama_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int32_t llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int32_t llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);
    LLAMA_API const char * llama_vicuna_core_system_prompt_default(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // The logger state is global so these functions are NOT thread safe.
    LLAMA_API void llama_log_get(ggml_log_callback * log_callback, void ** user_data);
    LLAMA_API void llama_log_set(ggml_log_callback   log_callback, void *  user_data);

    //
    // Performance utils
    //
    // NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct llama_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
    };

    struct llama_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);

    // NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);

    // print a breakdown of per-device memory use via LLAMA_LOG:
    LLAMA_API void llama_memory_breakdown_print(const struct llama_context * ctx);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct llama_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);

    LLAMA_API void llama_opt_epoch(
            struct llama_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#undef LLAMA_CPP_MEMBER_INIT

#endif // LLAMA_H
