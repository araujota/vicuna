#pragma once

#include "server-common.h"
#include "server-embedding-backend.h"

#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

enum server_emotive_block_kind {
    SERVER_EMOTIVE_BLOCK_USER_MESSAGE,
    SERVER_EMOTIVE_BLOCK_ASSISTANT_REASONING,
    SERVER_EMOTIVE_BLOCK_ASSISTANT_CONTENT,
    SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT,
};

struct server_emotive_vector {
    float epistemic_pressure = 0.0f;
    float confidence = 0.0f;
    float contradiction_pressure = 0.0f;
    float planning_clarity = 0.0f;
    float curiosity = 0.0f;
    float caution = 0.0f;
    float frustration = 0.0f;
    float satisfaction = 0.0f;
    float momentum = 0.0f;
    float stall = 0.0f;
    float semantic_novelty = 0.0f;
    float user_alignment = 0.0f;
    float runtime_trust = 0.0f;
    float runtime_failure_pressure = 0.0f;
};

struct server_emotive_delta {
    float d_epistemic_pressure = 0.0f;
    float d_confidence = 0.0f;
    float d_contradiction_pressure = 0.0f;
    float d_planning_clarity = 0.0f;
    float d_curiosity = 0.0f;
    float d_caution = 0.0f;
    float d_frustration = 0.0f;
    float d_satisfaction = 0.0f;
    float d_momentum = 0.0f;
    float d_stall = 0.0f;
    float d_semantic_novelty = 0.0f;
    float d_user_alignment = 0.0f;
    float d_runtime_trust = 0.0f;
    float d_runtime_failure_pressure = 0.0f;
    float negative_mass = 0.0f;
};

struct server_emotive_vad_trend {
    float d_valence = 0.0f;
    float d_arousal = 0.0f;
    float d_dominance = 0.0f;
};

struct server_emotive_style_guide {
    std::string tone_label;
    float warmth = 0.0f;
    float energy = 0.0f;
    float assertiveness = 0.0f;
    float empathy = 0.0f;
    float hedging = 0.0f;
    float directness = 0.0f;
    std::vector<std::string> prompt_hints;
};

struct server_emotive_vad {
    float valence = 0.0f;
    float arousal = 0.0f;
    float dominance = 0.0f;
    server_emotive_vad_trend trend;
    std::vector<std::string> labels;
    std::vector<std::string> dominant_dimensions;
    server_emotive_style_guide style_guide;
};

struct server_emotive_block_record {
    int32_t block_index = 0;
    server_emotive_block_kind kind = SERVER_EMOTIVE_BLOCK_USER_MESSAGE;
    std::string text;
    int32_t char_count = 0;
    server_emotive_vector moment;
    server_emotive_delta delta;
    server_emotive_vad vad;
    std::string embedding_mode;
    float semantic_similarity_to_user = 0.0f;
    float semantic_similarity_to_previous = 0.0f;
};

struct server_emotive_trace {
    bool valid = false;
    std::string trace_id;
    std::string model;
    std::vector<server_emotive_block_record> blocks;
    int32_t live_generation_start_block_index = 0;
    server_emotive_vector final_moment;
    server_emotive_vad final_vad;
    std::string embedding_mode;
    std::string estimator_version;
    bool provider_streamed = false;
    int32_t retained_block_count = 0;
    bool cognitive_replay = false;
    std::string cognitive_replay_entry_id;
    bool suppress_replay_admission = false;
    std::string mode_label;
    json final_policy = nullptr;
    json heuristic_retrieval = nullptr;
};

struct server_cognitive_replay_config {
    bool enabled = true;
    int32_t max_entries = 16;
    int32_t max_results = 8;
    float neg_mass_threshold = 0.02f;
    float valence_drop_threshold = 0.12f;
    float dominance_drop_threshold = 0.10f;
    int32_t persistence_blocks = 2;
    int32_t context_before = 1;
    int32_t context_after = 1;
    int32_t max_attempts = 2;
    int32_t idle_after_ms = 15000;
    int32_t poll_interval_ms = 500;
    float improvement_threshold = 0.08f;
};

enum server_cognitive_replay_status {
    SERVER_COGNITIVE_REPLAY_OPEN,
    SERVER_COGNITIVE_REPLAY_REVIEWING,
    SERVER_COGNITIVE_REPLAY_RESOLVED,
    SERVER_COGNITIVE_REPLAY_DEFERRED,
};

struct server_cognitive_replay_entry {
    std::string entry_id;
    server_cognitive_replay_status status = SERVER_COGNITIVE_REPLAY_OPEN;
    std::string source_trace_id;
    int64_t created_at_ms = 0;
    int64_t updated_at_ms = 0;
    int32_t trigger_block_index = 0;
    int32_t window_start_block_index = 0;
    int32_t window_end_block_index = 0;
    int32_t attempt_count = 0;
    int32_t max_attempts = 0;
    float negative_mass = 0.0f;
    float valence_drop = 0.0f;
    float dominance_drop = 0.0f;
    server_emotive_vad baseline_vad_before;
    server_emotive_vad baseline_vad_after;
    std::vector<server_emotive_block_record> window_blocks;
    std::string summary_excerpt;
    std::string last_error;
    std::string resolved_result_id;
};

struct server_cognitive_replay_comparison {
    float baseline_negative_mass = 0.0f;
    float replay_negative_mass = 0.0f;
    float baseline_valence = 0.0f;
    float replay_valence = 0.0f;
    float baseline_dominance = 0.0f;
    float replay_dominance = 0.0f;
    bool improved = false;
};

struct server_cognitive_replay_result {
    bool valid = false;
    std::string result_id;
    std::string entry_id;
    std::string trace_id;
    int64_t created_at_ms = 0;
    std::string reasoning_content;
    std::string content;
    server_cognitive_replay_comparison comparison;
    server_emotive_trace replay_trace;
};

struct server_heuristic_memory_config {
    bool enabled = true;
    std::string path = "vicuna-heuristic-memory.json";
    int32_t max_records = 64;
    int32_t top_k_semantic = 6;
    float semantic_threshold = 0.55f;
    float rerank_threshold = 0.72f;
    float semantic_weight = 0.55f;
    float struct_weight = 0.25f;
    float emotive_weight = 0.20f;
};

struct server_heuristic_object {
    std::string heuristic_id;
    std::string title;
    std::vector<std::string> task_types;
    std::vector<std::string> tool_names;
    std::vector<std::string> struct_tags;
    std::map<std::string, std::string> emotive_conditions;
    std::string semantic_trigger_text;
    std::string failure_mode;
    std::vector<std::string> evidence;
    std::vector<std::string> constraints;
    std::vector<std::string> preferred_actions;
    std::vector<std::string> action_ranking_rules;
    std::string mid_reasoning_correction;
    std::vector<std::string> applies_when;
    std::vector<std::string> avoid_when;
    float p_success = 0.5f;
    std::string calibration = "manual";
    std::string confidence_notes;
};

struct server_heuristic_bad_path_object {
    std::string object_id;
    std::string kind;
    std::string text;
    std::vector<std::string> struct_tags;
    std::vector<float> embedding;
};

struct server_heuristic_bad_signature {
    float negative_mass = 0.0f;
    float valence = 0.0f;
    float arousal = 0.0f;
    float dominance = 0.0f;
    std::vector<std::string> struct_tags;
};

struct server_heuristic_memory_record {
    std::string record_id;
    std::string entry_id;
    std::string result_id;
    std::string source_trace_id;
    int64_t created_at_ms = 0;
    std::string bad_path_text;
    std::string better_path_reasoning_content;
    std::string better_path_content;
    std::vector<server_heuristic_bad_path_object> bad_path_objects;
    server_heuristic_object heuristic;
    server_heuristic_bad_signature bad_signature;
};

struct server_heuristic_retrieval_decision {
    bool matched = false;
    std::string record_id;
    std::string heuristic_id;
    std::string query_text;
    float semantic_score = 0.0f;
    float struct_score = 0.0f;
    float emotive_score = 0.0f;
    float total_score = 0.0f;
    float threshold = 0.0f;
    int64_t created_at_ms = 0;
    std::vector<json> control_biases;
};

struct server_metacognitive_control_state {
    server_emotive_vector moment;
    server_emotive_vad vad;
    float ongoing_task_due = 0.0f;
    bool bridge_scoped = false;
    bool cognitive_replay = false;
    bool suppress_replay_admission = false;
    server_heuristic_retrieval_decision heuristic;
};

struct server_metacognitive_policy_decision {
    bool valid = false;
    std::string policy_version = "control_surface_v1";
    std::string selected_mode = "direct";
    std::string reasoning_depth = "short";
    float direct_score = 0.0f;
    float reflective_score = 0.0f;
    float tool_light_score = 0.0f;
    float tool_heavy_score = 0.0f;
    float background_defer_score = 0.0f;
    float reasoning_score = 0.0f;
    float tool_aggression = 0.0f;
    float interrupt_score = 0.0f;
    int32_t tool_parallelism_cap = 0;
    bool interrupt_allowed = false;
    bool replan_required = false;
    bool early_stop_ok = false;
    bool force_synthesis = false;
    std::vector<json> heuristic_biases;
    std::vector<std::string> prompt_hints;
};

struct server_emotive_runtime_config {
    bool enabled = true;
    int32_t block_max_chars = 320;
    int32_t max_blocks_per_turn = 128;
    int32_t max_turn_history = 8;
    bool degraded_mode_allowed = true;
    float vad_ema_alpha = 0.35f;
    server_embedding_backend_config embedding;
    server_cognitive_replay_config cognitive_replay;
    server_heuristic_memory_config heuristic_memory;
};

server_emotive_runtime_config server_emotive_runtime_config_from_env();

json server_emotive_trace_to_json(const server_emotive_trace & trace);

class server_emotive_runtime;

class server_emotive_turn_builder {
public:
    server_emotive_turn_builder(
            server_emotive_runtime & runtime,
            const std::string & model_name,
            bool cognitive_replay = false,
            const std::string & cognitive_replay_entry_id = std::string(),
            bool suppress_replay_admission = false,
            const std::string & mode_label = std::string());
    ~server_emotive_turn_builder();

    void add_user_message(const std::string & text);
    void add_replay_block(server_emotive_block_kind kind, const std::string & text);
    void observe_reasoning_delta(const std::string & text);
    void observe_content_delta(const std::string & text);
    void observe_runtime_event(const std::string & text);
    void mark_live_generation_start();
    void set_final_policy(json final_policy);
    void set_heuristic_retrieval(json heuristic_retrieval);
    bool has_current_state() const;
    server_emotive_vector current_moment() const;
    server_emotive_vad current_vad() const;
    server_emotive_trace finalize();

private:
    void append_text(server_emotive_block_kind kind, const std::string & text);
    void flush_pending();

    server_emotive_runtime & runtime_;
    std::string model_name_;
    server_emotive_block_kind pending_kind_;
    std::string pending_text_;
    bool has_pending_;
    std::vector<float> previous_embedding_;
    std::vector<float> user_anchor_embedding_;
    int32_t user_anchor_count_;
    server_emotive_vector previous_moment_;
    bool have_previous_moment_;
    server_emotive_vad previous_vad_;
    bool have_previous_vad_;
    std::vector<server_emotive_block_record> blocks_;
    int32_t live_generation_start_block_index_;
    bool live_generation_start_marked_;
    bool cognitive_replay_;
    std::string cognitive_replay_entry_id_;
    bool suppress_replay_admission_;
    std::string mode_label_;
    json final_policy_ = nullptr;
    json heuristic_retrieval_ = nullptr;
};

class server_emotive_runtime {
public:
    explicit server_emotive_runtime(const server_emotive_runtime_config & config);

    const server_emotive_runtime_config & config() const;

    server_emotive_block_record evaluate_block(
            server_emotive_block_kind kind,
            const std::string & text,
            int32_t block_index,
            const std::vector<float> & previous_embedding,
            const std::vector<float> & user_anchor_embedding,
            const server_emotive_vector * previous_moment,
            const server_emotive_vad * previous_vad) const;

    bool build_embedding(const std::string & text, std::vector<float> & out_embedding, std::string * out_mode = nullptr) const;
    void remember_trace(const server_emotive_trace & trace);

    json health_json() const;
    json latest_trace_json() const;
    json cognitive_replay_json() const;
    json heuristic_memory_json() const;
    bool try_claim_cognitive_replay_entry(server_cognitive_replay_entry * out_entry);
    void fail_cognitive_replay_entry(const std::string & entry_id, const std::string & error_message);
    bool get_cognitive_replay_resolution(
            const std::string & entry_id,
            server_cognitive_replay_entry * out_entry,
            server_cognitive_replay_result * out_result) const;
    void record_cognitive_replay_result(
            const std::string & entry_id,
            const std::string & reasoning_content,
            const std::string & content,
            const server_emotive_trace & replay_trace);
    bool store_heuristic_memory_record(
            const std::string & entry_id,
            const server_heuristic_object & heuristic,
            std::string * out_error = nullptr);
    server_heuristic_retrieval_decision retrieve_matching_heuristic(
            const std::string & query_text,
            const std::vector<std::string> & struct_tags,
            const server_emotive_vector * current_moment,
            const server_emotive_vad * current_vad,
            std::string * out_guidance = nullptr);
    server_metacognitive_policy_decision compute_control_policy(
            const server_metacognitive_control_state & state) const;

private:
    void load_heuristic_memory();
    server_emotive_runtime_config config_;
    mutable server_embedding_backend embedding_backend_;
    mutable std::mutex history_mutex_;
    std::deque<server_emotive_trace> latest_traces_;
    std::deque<server_cognitive_replay_entry> replay_entries_;
    std::deque<server_cognitive_replay_result> replay_results_;
    std::deque<server_heuristic_memory_record> heuristic_records_;
    server_heuristic_retrieval_decision last_heuristic_retrieval_;
    std::string heuristic_memory_error_;
};
