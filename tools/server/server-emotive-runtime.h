#pragma once

#include "server-common.h"
#include "server-embedding-backend.h"

#include <deque>
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
    server_emotive_vector final_moment;
    server_emotive_vad final_vad;
    std::string embedding_mode;
    std::string estimator_version;
    bool provider_streamed = false;
    int32_t retained_block_count = 0;
};

struct server_emotive_runtime_config {
    bool enabled = true;
    int32_t block_max_chars = 320;
    int32_t max_blocks_per_turn = 128;
    int32_t max_turn_history = 8;
    bool degraded_mode_allowed = true;
    float vad_ema_alpha = 0.35f;
    server_embedding_backend_config embedding;
};

server_emotive_runtime_config server_emotive_runtime_config_from_env();

json server_emotive_trace_to_json(const server_emotive_trace & trace);

class server_emotive_runtime;

class server_emotive_turn_builder {
public:
    server_emotive_turn_builder(server_emotive_runtime & runtime, const std::string & model_name);
    ~server_emotive_turn_builder();

    void add_user_message(const std::string & text);
    void observe_reasoning_delta(const std::string & text);
    void observe_content_delta(const std::string & text);
    void observe_runtime_event(const std::string & text);
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

private:
    server_emotive_runtime_config config_;
    mutable server_embedding_backend embedding_backend_;
    mutable std::mutex history_mutex_;
    std::deque<server_emotive_trace> latest_traces_;
};
