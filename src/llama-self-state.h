#pragma once

#include "llama.h"
#include "llama-vocab.h"

#include <array>
#include <cstdint>
#include <vector>

struct llama_self_register_definition {
    int32_t register_id = -1;
    const char * name = nullptr;
    int32_t family = LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR;
    float value_min = 0.0f;
    float value_max = 1.0f;
    float default_scalar_value = 0.0f;
    int32_t default_categorical_value = 0;
};

struct llama_self_register_value {
    float scalar_value = 0.0f;
    int32_t categorical_value = 0;
    float confidence = 1.0f;
    int64_t last_update_wall_ms = 0;
    int64_t last_update_monotonic_ms = 0;
    uint32_t source_mask = 0;
    uint32_t updater_version = 1;
    bool dirty = false;
};

struct llama_self_sketch_surface {
    int32_t id = -1;
    float priority = 0.0f;
    bool unresolved = false;
    int64_t last_update_monotonic_ms = -1;
    std::array<float, 32> sketch = {};
};

struct llama_self_working_memory_item {
    int32_t event_id = -1;
    int32_t role = LLAMA_SELF_STATE_EVENT_USER;
    uint32_t flags = 0;
    float salience = 0.0f;
    bool unresolved_question = false;
    bool tool_affordance_hint = false;
    int64_t admitted_monotonic_ms = -1;
    std::array<float, 32> sketch = {};
};

struct llama_self_memory_handle {
    int32_t handle_id = -1;
    int32_t kind = LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER;
    float priority = 0.0f;
    int64_t last_update_monotonic_ms = -1;
    uint32_t member_count = 0;
    std::array<float, 32> centroid = {};
};

struct llama_self_tool_job {
    int32_t job_id = -1;
    int32_t status = LLAMA_SELF_TOOL_JOB_IDLE;
    float importance = 0.0f;
    int64_t last_update_monotonic_ms = -1;
};

struct llama_self_trace_item {
    llama_self_state_time_point time_point = {};
    llama_self_state_event event = {};
    std::vector<llama_token> tokens;
};

class llama_self_state {
public:
    llama_self_state();

    bool refresh_time();
    bool set_time(llama_self_state_time_point time_point);
    bool get_datetime(llama_self_state_datetime * out_info) const;
    bool configure(const llama_self_state_params & params);

    int32_t register_count() const;
    bool get_register(int32_t register_id, llama_self_register_info * out_info) const;
    bool set_channel_state(int32_t channel_state);
    bool note_user_event();
    bool note_tool_event();
    bool note_emit_event();
    bool set_identity(const llama_token * tokens, size_t n_tokens);
    bool upsert_goal(int32_t goal_id, const llama_token * tokens, size_t n_tokens, float priority);
    bool upsert_commitment(int32_t commitment_id, const llama_token * tokens, size_t n_tokens, float priority, bool unresolved);
    int32_t goal_count() const;
    int32_t commitment_count() const;
    int32_t working_memory_count() const;
    bool upsert_memory_handle(int32_t handle_id, int32_t kind, const llama_token * tokens, size_t n_tokens, float priority);
    bool upsert_memory_handle_sketch(int32_t handle_id, int32_t kind, const std::array<float, 32> & centroid, float priority, uint32_t member_count);
    int32_t memory_handle_count() const;
    int32_t reactivation_count() const;
    bool get_reactivation(int32_t index, llama_self_reactivation_info * out_info) const;
    bool upsert_tool_job(int32_t job_id, int32_t status, float importance);
    bool get_tool_state(llama_self_tool_state_info * out_info) const;
    bool get_social_state(llama_self_social_state_info * out_info) const;
    int32_t trace_count() const;
    bool clear_trace();
    bool replay_trace(const llama_vocab * vocab, int32_t upto_count, int32_t override_channel);
    bool set_updater_program(const llama_self_updater_program & program);
    bool get_updater_program(llama_self_updater_program * out_program) const;
    size_t trace_export_size() const;
    bool trace_export(void * dst, size_t size) const;
    bool trace_import(const void * src, size_t size, bool replace_existing);
    bool evaluate_counterfactual(const llama_vocab * vocab, const llama_self_updater_program & program, int32_t upto_count, int32_t replay_channel, llama_self_counterfactual_result * out_result) const;
    bool build_prewrite_features(const llama_vocab * vocab, const llama_self_state_event & event, llama_self_state_feature_vector * out_features) const;
    bool apply_prewrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features);
    bool build_postwrite_features(const llama_vocab * vocab, const llama_self_state_event & event, llama_self_state_feature_vector * out_features) const;
    bool apply_postwrite(const llama_self_state_event & event, const llama_self_state_feature_vector & features);

    static const char * register_name(int32_t register_id);

private:
    bool ensure_time_initialized();
    bool apply_time_point(const llama_self_state_time_point & time_point, uint32_t source_mask);
    bool is_valid_time_point(const llama_self_state_time_point & time_point) const;
    bool build_features(const llama_vocab * vocab, const llama_self_state_event & event, bool postwrite, llama_self_state_feature_vector * out_features) const;
    float run_contradiction_head(float analytic_score, const llama_self_state_feature_vector & features) const;
    float run_uncertainty_head(float analytic_score, const llama_self_state_feature_vector & features) const;
    float run_broadcast_head(float analytic_score, const llama_self_state_feature_vector & features) const;
    float max_similarity(const std::vector<llama_self_sketch_surface> & surfaces, const std::array<float, 32> & sketch, bool unresolved_only) const;
    void working_memory_stats(const std::array<float, 32> & sketch, float * out_top_similarity, float * out_variance) const;
    void memory_handle_stats(const std::array<float, 32> & sketch, float * out_top_similarity, float * out_variance) const;
    void upsert_surface(std::vector<llama_self_sketch_surface> & surfaces, int32_t id, const std::array<float, 32> & sketch, float priority, bool unresolved);
    void admit_working_memory(const llama_self_state_event & event, const std::array<float, 32> & sketch, const llama_self_state_feature_vector & features);
    void update_reactivation_priorities(const std::array<float, 32> & sketch, float memory_write_pressure);
    void refresh_tool_surface(uint32_t source_mask);
    void reset_dynamic_state_preserve_static();
    void append_trace(const llama_self_state_event & event);
    void bridge_working_memory_to_handles(const std::array<float, 32> & sketch, float salience);
    void update_social_state(const llama_self_state_event & event, const llama_self_state_feature_vector & features);

    void recompute_time_surface(uint32_t source_mask);
    void update_scalar_register(int32_t register_id, float value, uint32_t source_mask);
    void update_categorical_register(int32_t register_id, int32_t value, uint32_t source_mask);
    float current_scalar_register(int32_t register_id) const;
    void blend_scalar_register(int32_t register_id, float target, float gain, uint32_t source_mask);
    bool validate_updater_program(const llama_self_updater_program & program) const;
    bool apply_register_update_rules(uint32_t phase_mask, const llama_self_state_event & event, const llama_self_state_feature_vector & features, uint32_t source_mask);
    float updater_feature_value(int32_t feature_id, const llama_self_state_event & event, const llama_self_state_feature_vector & features) const;

    static const llama_self_register_definition * get_definition(int32_t register_id);
    static std::array<llama_self_register_definition, LLAMA_SELF_REGISTER_COUNT> build_definitions();

private:
    std::array<llama_self_register_definition, LLAMA_SELF_REGISTER_COUNT> definitions;
    std::array<llama_self_register_value, LLAMA_SELF_REGISTER_COUNT> registers;
    llama_self_state_datetime datetime = {};
    llama_self_state_params params = {};
    llama_self_updater_program updater_program = {};

    int32_t channel_state = LLAMA_SELF_STATE_CHANNEL_WAITING;

    int64_t session_start_wall_ms = -1;
    int64_t session_start_monotonic_ms = -1;
    int64_t last_user_monotonic_ms = -1;
    int64_t last_tool_monotonic_ms = -1;
    int64_t last_emit_monotonic_ms = -1;
    int64_t social_last_update_monotonic_ms = -1;
    int32_t next_working_memory_event_id = 1;
    int32_t social_user_turn_count = 0;
    int32_t social_system_turn_count = 0;
    bool has_explicit_time = false;
    bool has_previous_event_sketch = false;
    std::array<float, 32> previous_event_sketch = {};
    std::array<float, 32> identity_sketch = {};
    bool has_identity_sketch = false;
    float social_familiarity = 0.0f;
    float social_trust = 0.5f;
    float social_reciprocity = 0.5f;
    std::vector<llama_self_sketch_surface> goals;
    std::vector<llama_self_sketch_surface> commitments;
    std::vector<llama_self_working_memory_item> working_memory;
    std::vector<llama_self_memory_handle> memory_handles;
    std::vector<llama_self_reactivation_info> reactivation_priorities;
    std::vector<llama_self_tool_job> tool_jobs;
    std::vector<llama_self_trace_item> trace_items;
};
