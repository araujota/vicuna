#include "get-model.h"
#include "llama.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <array>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokenize_or_die(const llama_vocab * vocab, const std::string & text) {
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "failed to tokenize self-state test prompt\n");
        std::exit(1);
    }
    return tokens;
}

static bool find_model_extension_by_key(
        llama_context * ctx,
        const std::string & key,
        llama_self_model_extension_info * out_info) {
    if (!ctx || !out_info) {
        return false;
    }

    const int32_t count = llama_self_state_model_extension_count(ctx);
    for (int32_t i = 0; i < count; ++i) {
        llama_self_model_extension_info info = {};
        if (llama_self_state_get_model_extension(ctx, i, &info) != 0) {
            continue;
        }
        if (std::string(info.key) == key) {
            *out_info = info;
            return true;
        }
    }

    return false;
}

static bool contradiction_head_callback(
        const llama_self_state_feature_vector * features,
        float * out_score,
        void * /*user_data*/) {
    if (!features || !out_score) {
        return false;
    }

    *out_score = 0.85f;
    return true;
}

static bool uncertainty_head_callback(
        const llama_self_state_feature_vector * features,
        float * out_score,
        void * /*user_data*/) {
    if (!features || !out_score) {
        return false;
    }

    *out_score = 0.75f;
    return true;
}

static bool broadcast_head_callback(
        const llama_self_state_feature_vector * features,
        float * out_score,
        void * /*user_data*/) {
    if (!features || !out_score) {
        return false;
    }

    *out_score = 0.65f;
    return true;
}

static void configure_small_buckets(llama_past_lora_params & params) {
    params.enabled = true;

    const std::array<uint64_t, LLAMA_MEMORY_LORA_BUCKET_COUNT> periods = { 100, 200, 300, 400, 0 };
    const std::array<uint64_t, LLAMA_MEMORY_LORA_BUCKET_COUNT> half_lives = { 500, 600, 700, 800, 900 };
    const std::array<float, LLAMA_MEMORY_LORA_BUCKET_COUNT> base_scales = { 0.90f, 0.70f, 0.50f, 0.30f, 0.10f };

    for (int i = 0; i < LLAMA_MEMORY_LORA_BUCKET_COUNT; ++i) {
        params.host_memory_ratio[i] = 1.0f;
        params.device_memory_ratio[i] = 1.0f;
        params.min_rank[i] = 1;
        params.max_rank[i] = 2;
        params.condensation_period_us[i] = periods[i];
        params.decay_half_life_us[i] = half_lives[i];
        params.base_scale[i] = base_scales[i];
    }
}

struct mock_supermemory_server {
    httplib::Server server;
    std::thread thread;
    std::mutex mutex;
    std::atomic<int> profile_calls = 0;
    std::atomic<int> archive_calls = 0;
    std::string last_profile_body;
    std::string last_archive_body;
    int port = -1;

    bool start() {
        server.Post("/v4/profile", [this](const httplib::Request & req, httplib::Response & res) {
            ++profile_calls;
            {
                std::lock_guard<std::mutex> lock(mutex);
                last_profile_body = req.body;
            }
            res.status = 200;
            res.set_content(json({
                {"profile", {
                    {"static", json::array({"Prefers explicit runtime policy", "Uses hard memory"})},
                    {"dynamic", json::array({"Recently perturbed by contradiction-heavy prompts"})},
                }},
                {"searchResults", {
                    {"results", json::array({
                        {
                            {"id", "mem_alpha"},
                            {"memory", "Earlier repair attempt referenced the strict memory runtime."},
                            {"similarity", 0.91},
                            {"title", "repair-memory"},
                            {"metadata", {
                                {"kind", "self_model_fragment"},
                                {"domain", "epistemic"},
                                {"sourceRole", LLAMA_SELF_STATE_EVENT_SYSTEM},
                                {"sourceChannel", LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY},
                                {"sourceToolKind", LLAMA_TOOL_KIND_HARD_MEMORY_QUERY},
                                {"flags", LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_GAIN},
                                {"importance", 0.86},
                                {"confidence", 0.79},
                                {"gainBias", 0.88},
                                {"allostaticRelevance", 0.12},
                                {"tags", json::array({"repair", "self_model", "epistemic"})},
                            }},
                        },
                        {
                            {"id", "chunk_beta"},
                            {"chunk", "Older context from last week favored a narrower repository query."},
                            {"similarity", 0.83},
                            {"title", "last-week-query"},
                            {"metadata", {
                                {"kind", "user_model"},
                                {"domain", "user_outcome"},
                                {"sourceRole", LLAMA_SELF_STATE_EVENT_USER},
                                {"sourceChannel", LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY},
                                {"sourceToolKind", LLAMA_TOOL_KIND_HARD_MEMORY_QUERY},
                                {"flags", LLAMA_HARD_MEMORY_PRIMITIVE_AFFECT_GAIN},
                                {"importance", 0.63},
                                {"confidence", 0.71},
                                {"gainBias", 0.52},
                                {"allostaticRelevance", 0.34},
                                {"tags", json::array({"user_model", "last_week"})},
                            }},
                        }
                    })}
                }}
            }).dump(), "application/json");
        });

        server.Post("/v4/memories", [this](const httplib::Request & req, httplib::Response & res) {
            ++archive_calls;
            {
                std::lock_guard<std::mutex> lock(mutex);
                last_archive_body = req.body;
            }
            res.status = 201;
            res.set_content(json({
                {"documentId", "doc_delta"},
                {"memories", json::array({
                    json({
                        {"id", "mem_delta"},
                        {"memory", "archived perturbation"},
                        {"isStatic", false}
                    })
                })}
            }).dump(), "application/json");
        });

        port = server.bind_to_any_port("127.0.0.1");
        if (port <= 0) {
            return false;
        }

        thread = std::thread([this]() {
            server.listen_after_bind();
        });
        return true;
    }

    void stop() {
        server.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }
};

static bool verify_runtime_persistence_surfaces(
        llama_context * ctx,
        llama_model * model,
        const llama_context_params & cparams,
        const llama_self_updater_program & program,
        int32_t expected_trace_count) {
    llama_self_model_extension_update persisted_extension = llama_self_model_extension_default_update();
    persisted_extension.source = LLAMA_SELF_MODEL_EXTENSION_SOURCE_EVENT_FEEDBACK;
    persisted_extension.kind = LLAMA_SELF_MODEL_EXTENSION_SCALAR_PARAM;
    persisted_extension.domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_RECOVERY;
    persisted_extension.lifecycle_stage = LLAMA_SELF_MODEL_EXTENSION_STAGE_ALLOSTATIC;
    persisted_extension.flags = LLAMA_SELF_MODEL_EXTENSION_FLAG_ACTIVE |
            LLAMA_SELF_MODEL_EXTENSION_FLAG_DISCOVERED |
            LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN |
            LLAMA_SELF_MODEL_EXTENSION_FLAG_HAS_DESIRED_STATE |
            LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS;
    persisted_extension.support_count = 4;
    persisted_extension.value = 0.72f;
    persisted_extension.desired_value = 0.85f;
    persisted_extension.desired_value_min = 0.80f;
    persisted_extension.desired_value_max = 0.90f;
    persisted_extension.confidence = 0.81f;
    persisted_extension.salience = 0.66f;
    persisted_extension.gain_weight = 0.40f;
    persisted_extension.allostatic_weight = 0.76f;
    persisted_extension.surprise_score = 0.61f;
    persisted_extension.relevance_score = 0.73f;
    persisted_extension.admission_score = 0.69f;
    persisted_extension.permanence_score = 0.88f;
    persisted_extension.stability_score = 0.83f;
    persisted_extension.allostatic_eligibility = 0.91f;
    std::snprintf(persisted_extension.key, sizeof(persisted_extension.key), "%s", "persistence.discovered_register");
    std::snprintf(persisted_extension.label, sizeof(persisted_extension.label), "%s", "Discovered persistence register");
    std::snprintf(persisted_extension.content, sizeof(persisted_extension.content), "%s", "Recovered from runtime snapshot");

    llama_self_updater_program persisted_program = program;
    persisted_program.repair_emit_threshold += 0.05f;
    persisted_program.repair_admission_floor += 0.02f;

    llama_bash_tool_config persisted_bash = llama_bash_tool_default_config();
    persisted_bash.enabled = true;
    persisted_bash.reject_shell_metacharacters = true;
    persisted_bash.cpu_time_limit_secs = 4;
    persisted_bash.max_child_processes = 4;
    persisted_bash.max_open_files = 24;
    persisted_bash.max_file_size_bytes = 4096;
    std::snprintf(persisted_bash.allowed_commands, sizeof(persisted_bash.allowed_commands), "%s", "pwd,rg");
    std::snprintf(persisted_bash.blocked_patterns, sizeof(persisted_bash.blocked_patterns), "%s", "rm -rf,:(){:|:&};:");

    llama_hard_memory_config persisted_hard_memory = llama_hard_memory_default_config();
    persisted_hard_memory.enabled = true;
    persisted_hard_memory.timeout_ms = 4321;
    persisted_hard_memory.max_results = 3;
    persisted_hard_memory.query_threshold = 0.42f;
    std::snprintf(persisted_hard_memory.base_url, sizeof(persisted_hard_memory.base_url), "%s", "http://127.0.0.1:9999");
    std::snprintf(persisted_hard_memory.auth_token, sizeof(persisted_hard_memory.auth_token), "%s", "persist-token");
    std::snprintf(persisted_hard_memory.container_tag, sizeof(persisted_hard_memory.container_tag), "%s", "persist-container");
    std::snprintf(persisted_hard_memory.runtime_identity, sizeof(persisted_hard_memory.runtime_identity), "%s", "persist-runtime");

    if (llama_self_state_set_updater_program(ctx, persisted_program) != 0 ||
        llama_self_state_upsert_model_extension(ctx, persisted_extension) != 0 ||
        llama_bash_tool_configure(ctx, &persisted_bash) != 0 ||
        llama_hard_memory_configure(ctx, persisted_hard_memory) != 0) {
        std::fprintf(stderr, "failed to seed runtime persistence surfaces\n");
        return false;
    }

    const size_t persisted_trace_size = llama_self_state_trace_export_size(ctx);
    std::vector<uint8_t> persisted_trace(persisted_trace_size);
    if (persisted_trace_size == 0 ||
        llama_self_state_trace_export(ctx, persisted_trace.data(), persisted_trace.size()) != 0) {
        std::fprintf(stderr, "failed to export persisted trace\n");
        return false;
    }

    llama_context * restored_ctx = llama_init_from_model(model, cparams);
    if (!restored_ctx) {
        std::fprintf(stderr, "failed to create restored runtime context\n");
        return false;
    }

    if (llama_self_state_set_updater_program(restored_ctx, persisted_program) != 0 ||
        llama_bash_tool_configure(restored_ctx, &persisted_bash) != 0 ||
        llama_hard_memory_configure(restored_ctx, persisted_hard_memory) != 0 ||
        llama_self_state_trace_import(restored_ctx, persisted_trace.data(), persisted_trace.size(), true) != 0 ||
        llama_self_state_replay_trace(restored_ctx, -1) != 0 ||
        llama_self_state_upsert_model_extension(restored_ctx, persisted_extension) != 0) {
        std::fprintf(stderr, "failed to replay runtime persistence snapshot\n");
        llama_free(restored_ctx);
        return false;
    }

    llama_self_updater_program restored_program = {};
    llama_bash_tool_config restored_bash = {};
    llama_hard_memory_config restored_hard_memory = {};
    llama_self_model_extension_info restored_extension = {};
    const int32_t expected_restored_extension_count = llama_self_state_model_extension_count(ctx);
    const bool restored_ok =
            llama_self_state_get_updater_program(restored_ctx, &restored_program) == 0 &&
            llama_bash_tool_get_config(restored_ctx, &restored_bash) == 0 &&
            llama_hard_memory_get_config(restored_ctx, &restored_hard_memory) == 0 &&
            llama_self_state_model_extension_count(restored_ctx) == expected_restored_extension_count &&
            find_model_extension_by_key(restored_ctx, "persistence.discovered_register", &restored_extension) &&
            restored_program.repair_emit_threshold == persisted_program.repair_emit_threshold &&
            restored_program.repair_admission_floor == persisted_program.repair_admission_floor &&
            restored_bash.enabled &&
            restored_bash.reject_shell_metacharacters &&
            restored_bash.cpu_time_limit_secs == persisted_bash.cpu_time_limit_secs &&
            std::string(restored_bash.allowed_commands) == "pwd,rg" &&
            restored_hard_memory.enabled &&
            restored_hard_memory.timeout_ms == persisted_hard_memory.timeout_ms &&
            restored_hard_memory.max_results == persisted_hard_memory.max_results &&
            std::string(restored_hard_memory.container_tag) == "persist-container" &&
            std::string(restored_extension.key) == "persistence.discovered_register" &&
            (restored_extension.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_DISCOVERED) != 0 &&
            restored_extension.lifecycle_stage == LLAMA_SELF_MODEL_EXTENSION_STAGE_ALLOSTATIC &&
            restored_extension.support_count == 4 &&
            restored_extension.desired_value_min >= 0.79f &&
            restored_extension.desired_value_max >= 0.89f &&
            restored_extension.permanence_score >= 0.87f &&
            restored_extension.allostatic_eligibility >= 0.90f &&
            llama_self_state_trace_count(restored_ctx) == expected_trace_count;
    if (!restored_ok) {
        std::fprintf(stderr, "runtime persistence surfaces did not survive snapshot replay\n");
        llama_free(restored_ctx);
        return false;
    }

    llama_free(restored_ctx);
    return true;
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (llama_self_state_register_count(ctx) != LLAMA_SELF_REGISTER_COUNT) {
        std::fprintf(stderr, "unexpected self-state register count\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_state_datetime datetime = {};
    if (llama_self_state_get_datetime(ctx, &datetime) != 0) {
        std::fprintf(stderr, "failed to query self-state datetime\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (datetime.wall_clock_ms <= 0 || datetime.monotonic_ms <= 0 ||
        datetime.local_hour < 0 || datetime.local_hour > 23 ||
        datetime.local_minute < 0 || datetime.local_minute > 59) {
        std::fprintf(stderr, "unexpected default self-state datetime\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_register_info register_info = {};
    if (llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_TIME_PHASE, &register_info) != 0 ||
        register_info.family != LLAMA_SELF_REGISTER_FAMILY_BOUNDED_SCALAR ||
        register_info.scalar_value < 0.0f || register_info.scalar_value > 1.0f) {
        std::fprintf(stderr, "unexpected time-phase register\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!llama_self_state_register_name(LLAMA_SELF_REGISTER_CHANNEL_STATE)) {
        std::fprintf(stderr, "missing channel-state register name\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (!llama_self_state_register_name(LLAMA_SELF_REGISTER_ANSWERABILITY)) {
        std::fprintf(stderr, "missing expanded self-state register name\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_self_state_time_point t0 = {
        /*.wall_clock_ms =*/ 1700000000000,
        /*.monotonic_ms =*/ 1000,
        /*.timezone_offset_minutes =*/ -360,
    };

    if (llama_self_state_set_time(ctx, t0) != 0) {
        std::fprintf(stderr, "failed to apply deterministic time point\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_note_user_event(ctx) != 0 ||
        llama_self_state_note_tool_event(ctx) != 0 ||
        llama_self_state_note_emit_event(ctx) != 0) {
        std::fprintf(stderr, "failed to apply self-state event anchors\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_self_state_time_point t1 = {
        /*.wall_clock_ms =*/ 1700000300000,
        /*.monotonic_ms =*/ 301000,
        /*.timezone_offset_minutes =*/ -360,
    };

    if (llama_self_state_set_time(ctx, t1) != 0 ||
        llama_self_state_get_datetime(ctx, &datetime) != 0) {
        std::fprintf(stderr, "failed to advance self-state time\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (datetime.session_age_ms != 300000 ||
        datetime.delta_since_last_user_ms != 300000 ||
        datetime.delta_since_last_tool_event_ms != 300000 ||
        datetime.delta_since_last_emit_ms != 300000) {
        std::fprintf(stderr, "unexpected elapsed self-state deltas\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_TOOL_SALIENCE, &register_info) != 0 ||
        register_info.scalar_value < 0.49f || register_info.scalar_value > 0.51f ||
        (register_info.source_mask & LLAMA_SELF_SOURCE_TOOL_EVENT) == 0) {
        std::fprintf(stderr, "unexpected tool-salience register\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_set_channel_state(ctx, LLAMA_SELF_STATE_CHANNEL_ACTIVE) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_CHANNEL_STATE, &register_info) != 0 ||
        register_info.family != LLAMA_SELF_REGISTER_FAMILY_CATEGORICAL ||
        register_info.categorical_value != LLAMA_SELF_STATE_CHANNEL_ACTIVE) {
        std::fprintf(stderr, "unexpected channel-state register\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_self_state_time_point invalid_time = {
        /*.wall_clock_ms =*/ 1700000299000,
        /*.monotonic_ms =*/ 300999,
        /*.timezone_offset_minutes =*/ -360,
    };

    if (llama_self_state_set_time(ctx, invalid_time) == 0) {
        std::fprintf(stderr, "accepted non-monotonic self-state time\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> identity_tokens = tokenize_or_die(vocab, "I am Vicuna and I maintain the strict memory runtime.");
    const std::vector<llama_token> goal_tokens = tokenize_or_die(vocab, "Finish the strict memory runtime implementation.");
    const std::vector<llama_token> commitment_tokens = tokenize_or_die(vocab, "I will finish the strict memory runtime implementation.");
    const std::vector<llama_token> memory_handle_tokens = tokenize_or_die(vocab, "strict memory runtime implementation working memory cluster");

    if (llama_self_state_set_identity(ctx, identity_tokens.data(), identity_tokens.size()) != 0 ||
        llama_self_state_upsert_goal(ctx, 7, goal_tokens.data(), goal_tokens.size(), 1.0f) != 0 ||
        llama_self_state_upsert_commitment(ctx, 11, commitment_tokens.data(), commitment_tokens.size(), 1.0f, true) != 0 ||
        llama_self_state_upsert_memory_handle(ctx, 17, LLAMA_SELF_MEMORY_HANDLE_WORKING_MEMORY_CLUSTER, memory_handle_tokens.data(), memory_handle_tokens.size(), 0.9f) != 0 ||
        llama_self_state_goal_count(ctx) != 1 ||
        llama_self_state_commitment_count(ctx) != 1 ||
        llama_self_state_memory_handle_count(ctx) != 1 ||
        llama_self_state_working_memory_count(ctx) != 0) {
        std::fprintf(stderr, "failed to configure self-state retrieval surfaces\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> commitment_event_tokens = tokenize_or_die(vocab, "I will finish the strict memory runtime implementation today.");
    llama_self_state_event commitment_event = {
        /*.tokens =*/ commitment_event_tokens.data(),
        /*.n_tokens =*/ commitment_event_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
        /*.decoder_entropy =*/ 0.4f,
        /*.decoder_top_margin =*/ 0.7f,
    };

    llama_self_state_feature_vector commitment_features = {};
    llama_self_state_feature_vector commitment_postwrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &commitment_event, &commitment_features) != 0 ||
        commitment_features.goal_top_similarity <= 0.05f ||
        commitment_features.commitment_top_similarity <= 0.05f ||
        commitment_features.identity_similarity <= 0.05f ||
        llama_self_state_apply_prewrite(ctx, &commitment_event, &commitment_features) != 0 ||
        llama_self_state_build_postwrite_features(ctx, &commitment_event, &commitment_postwrite) != 0 ||
        llama_self_state_apply_postwrite(ctx, &commitment_event, &commitment_postwrite) != 0 ||
        llama_self_state_working_memory_count(ctx) != 1 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_GOAL_RELEVANCE, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f) {
        std::fprintf(stderr, "self-state retrieval surfaces did not influence updates\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> related_event_tokens = tokenize_or_die(vocab, "strict memory runtime implementation");
    llama_self_state_event related_event = {
        /*.tokens =*/ related_event_tokens.data(),
        /*.n_tokens =*/ related_event_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ 0,
        /*.decoder_entropy =*/ 0.2f,
        /*.decoder_top_margin =*/ 0.8f,
    };

    llama_self_state_feature_vector retrieval_features = {};
    if (llama_self_state_build_prewrite_features(ctx, &related_event, &retrieval_features) != 0 ||
        retrieval_features.working_memory_top_similarity <= 0.05f ||
        retrieval_features.memory_handle_top_similarity <= 0.05f) {
        std::fprintf(stderr, "working-memory retrieval features were not populated\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_active_lora_params active = llama_active_lora_default_params();
    active.enabled = true;
    active.host_memory_ratio = 1.0f;
    active.device_memory_ratio = 1.0f;
    active.min_rank = 1;
    active.max_rank = 2;
    active.max_updates_before_rollover = 1;
    active.gain_max = 0.25f;

    llama_past_lora_params past = llama_past_lora_default_params();
    configure_small_buckets(past);
    const std::vector<llama_token> frozen_tokens = tokenize_or_die(vocab, "A frozen memory test span that should push the active adapter over its rollover boundary.");

    if (llama_active_lora_init(ctx, active) != 0 ||
        llama_past_lora_init(ctx, past) != 0 ||
        llama_active_lora_ingest(ctx, frozen_tokens.data(), frozen_tokens.size()) != 0 ||
        llama_past_lora_tick(ctx, 100) != 0 ||
        llama_self_state_memory_handle_count(ctx) < 2) {
        std::fprintf(stderr, "failed to bridge frozen LoRA buckets into self-state handles\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> user_tokens = tokenize_or_die(vocab, "I am not sure this answer is correct.");
    llama_self_state_event user_event = {
        /*.tokens =*/ user_tokens.data(),
        /*.n_tokens =*/ user_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
        /*.decoder_entropy =*/ 2.5f,
        /*.decoder_top_margin =*/ 0.20f,
    };

    llama_self_state_feature_vector prewrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &user_event, &prewrite) != 0 ||
        prewrite.novelty < 0.0f || prewrite.novelty > 1.0f ||
        prewrite.uncertainty_score <= 0.0f ||
        prewrite.contradiction_score <= 0.0f) {
        std::fprintf(stderr, "unexpected prewrite feature vector\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_apply_prewrite(ctx, &user_event, &prewrite) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_UNCERTAINTY, &register_info) != 0 ||
        register_info.scalar_value < 0.0f || register_info.scalar_value > 1.0f) {
        std::fprintf(stderr, "failed to apply prewrite features\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_state_feature_vector postwrite = {};
    if (llama_self_state_build_postwrite_features(ctx, &user_event, &postwrite) != 0 ||
        llama_self_state_apply_postwrite(ctx, &user_event, &postwrite) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_BROADCAST_PRESSURE, &register_info) != 0 ||
        register_info.scalar_value < 0.0f || register_info.scalar_value > 1.0f) {
        std::fprintf(stderr, "failed to apply postwrite features\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_state_params params = llama_self_state_default_params();
    params.enable_learned_contradiction_head = true;
    params.enable_learned_uncertainty_head = true;
    params.enable_learned_broadcast_head = true;
    params.contradiction_head_callback = contradiction_head_callback;
    params.uncertainty_head_callback = uncertainty_head_callback;
    params.broadcast_head_callback = broadcast_head_callback;

    if (llama_self_state_configure(ctx, params) != 0) {
        std::fprintf(stderr, "failed to configure learned self-state heads\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> tool_tokens = tokenize_or_die(vocab, "Tool failed with error: cannot open file.");
    llama_self_state_event tool_event = {
        /*.tokens =*/ tool_tokens.data(),
        /*.n_tokens =*/ tool_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_TOOL_FAILED,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 0.0f,
    };

    llama_self_state_feature_vector learned_prewrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &tool_event, &learned_prewrite) != 0 ||
        learned_prewrite.contradiction_score < 0.84f ||
        learned_prewrite.uncertainty_score < 0.74f) {
        std::fprintf(stderr, "learned heads did not influence feature vector\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_apply_prewrite(ctx, &tool_event, &learned_prewrite) != 0 ||
        llama_self_state_build_postwrite_features(ctx, &tool_event, &postwrite) != 0 ||
        postwrite.broadcast_pressure_hint < 0.64f ||
        llama_self_state_apply_postwrite(ctx, &tool_event, &postwrite) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_CONTRADICTION, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f) {
        std::fprintf(stderr, "failed learned-head register update path\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> preference_tokens = tokenize_or_die(
            vocab,
            "Be direct.\n- Show the failing path\n- Fix it\nDo not ask extra questions.");
    llama_self_state_event preference_event = {
        /*.tokens =*/ preference_tokens.data(),
        /*.n_tokens =*/ preference_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
        /*.decoder_entropy =*/ 0.15f,
        /*.decoder_top_margin =*/ 0.92f,
    };
    llama_self_state_feature_vector preference_prewrite = {};
    llama_self_state_feature_vector preference_postwrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &preference_event, &preference_prewrite) != 0 ||
        llama_self_state_apply_prewrite(ctx, &preference_event, &preference_prewrite) != 0 ||
        llama_self_state_build_postwrite_features(ctx, &preference_event, &preference_postwrite) != 0 ||
        llama_self_state_apply_postwrite(ctx, &preference_event, &preference_postwrite) != 0) {
        std::fprintf(stderr, "failed to apply user-preference rhetorical event\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_upsert_tool_job(ctx, 3, LLAMA_SELF_TOOL_JOB_PENDING, 0.6f) != 0 ||
        llama_self_state_upsert_tool_job(ctx, 4, LLAMA_SELF_TOOL_JOB_RUNNING, 0.8f) != 0) {
        std::fprintf(stderr, "failed to update tool jobs\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_tool_state_info tool_state = {};
    if (llama_self_state_get_tool_state(ctx, &tool_state) != 0 ||
        tool_state.pending_jobs != 1 ||
        tool_state.running_jobs != 1 ||
        tool_state.readiness >= 0.5f) {
        std::fprintf(stderr, "unexpected tool-state surface\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_build_postwrite_features(ctx, &tool_event, &postwrite) != 0 ||
        postwrite.tool_pending_pressure <= 0.0f ||
        postwrite.tool_readiness_score >= 0.5f) {
        std::fprintf(stderr, "tool lifecycle features were not populated\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const std::vector<llama_token> discovered_tokens = tokenize_or_die(
            vocab,
            "I am still unsure the strict memory runtime implementation is correct because the latest result contradicts the current repair path and the answer looks wrong.");
    llama_self_state_event discovered_event = {
        /*.tokens =*/ discovered_tokens.data(),
        /*.n_tokens =*/ discovered_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
        /*.decoder_entropy =*/ 2.8f,
        /*.decoder_top_margin =*/ 0.08f,
    };
    llama_self_state_feature_vector discovered_prewrite = {};
    llama_self_state_feature_vector discovered_postwrite = {};
    if (llama_self_state_build_prewrite_features(ctx, &discovered_event, &discovered_prewrite) != 0 ||
        discovered_prewrite.goal_top_similarity <= 0.05f ||
        discovered_prewrite.uncertainty_score <= 0.0f ||
        discovered_prewrite.contradiction_score <= 0.0f ||
        llama_self_state_apply_prewrite(ctx, &discovered_event, &discovered_prewrite) != 0 ||
        llama_self_state_build_postwrite_features(ctx, &discovered_event, &discovered_postwrite) != 0 ||
        llama_self_state_apply_postwrite(ctx, &discovered_event, &discovered_postwrite) != 0) {
        std::fprintf(stderr, "failed to apply discovered-state stress event\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_social_state_info social_state = {};
    if (llama_self_state_get_social_state(ctx, &social_state) != 0 ||
        social_state.familiarity <= 0.0f ||
        social_state.trust <= 0.0f || social_state.trust > 1.0f ||
        social_state.reciprocity <= 0.0f ||
        social_state.bond_strength <= 0.0f ||
        social_state.user_turn_count < 2 ||
        social_state.system_turn_count < 1) {
        std::fprintf(stderr, "social relationship state was not maintained as persistent scalars\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_model_state_info model_state = {};
    if (llama_self_state_get_model_state(ctx, &model_state) != 0 ||
        model_state.horizon_count != LLAMA_SELF_HORIZON_COUNT ||
        !model_state.belief_summary.valid ||
        model_state.belief_slot_count <= 0 ||
        !model_state.forecast.valid ||
        model_state.horizons[LLAMA_SELF_HORIZON_INSTANT].epistemic.answerability <= 0.0f ||
        model_state.horizons[LLAMA_SELF_HORIZON_SHORT].efficiency.expected_steps_remaining <= 0.0f ||
        model_state.horizons[LLAMA_SELF_HORIZON_LONG].user_outcome.satisfaction_estimate < 0.0f ||
        model_state.horizons[LLAMA_SELF_HORIZON_INSTANT].self_improvement.update_worthiness <= 0.0f ||
        model_state.horizons[LLAMA_SELF_HORIZON_INSTANT].user_preference.directness_preference <= 0.0f ||
        model_state.horizons[LLAMA_SELF_HORIZON_INSTANT].user_preference.structure_preference <= 0.0f ||
        model_state.horizons[LLAMA_SELF_HORIZON_INSTANT].user_preference.simulator_readiness <= 0.0f ||
        model_state.belief_summary.known_care_uncertainty < 0.0f ||
        model_state.belief_summary.belief_confidence < 0.0f) {
        std::fprintf(stderr, "expanded self-model state was not populated\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const int32_t baseline_extension_count = llama_self_state_model_extension_count(ctx);
    if (baseline_extension_count <= 0 ||
        model_state.extension_summary.discovered_count <= 0 ||
        model_state.extension_summary.transient_count <= 0 ||
        model_state.extension_summary.mean_admission <= 0.0f ||
        model_state.extension_summary.context_activation <= 0.0f) {
        std::fprintf(stderr, "discovered-state admission did not populate the lifecycle registry\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_model_extension_info discovered_context = {};
    bool found_discovered_context = false;
    llama_self_model_extension_info discovered_scalar = {};
    bool found_discovered_scalar = false;
    for (int32_t i = 0; i < baseline_extension_count; ++i) {
        llama_self_model_extension_info info = {};
        if (llama_self_state_get_model_extension(ctx, i, &info) != 0) {
            continue;
        }
        if (!found_discovered_context &&
            info.source == LLAMA_SELF_MODEL_EXTENSION_SOURCE_EVENT_FEEDBACK &&
            info.kind == LLAMA_SELF_MODEL_EXTENSION_MEMORY_CONTEXT &&
            (info.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_DISCOVERED) != 0) {
            discovered_context = info;
            found_discovered_context = true;
        }
        if (!found_discovered_scalar &&
            info.source == LLAMA_SELF_MODEL_EXTENSION_SOURCE_EVENT_FEEDBACK &&
            info.kind == LLAMA_SELF_MODEL_EXTENSION_SCALAR_PARAM &&
            (info.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_DISCOVERED) != 0) {
            discovered_scalar = info;
            found_discovered_scalar = true;
        }
    }
    if (!found_discovered_context ||
        discovered_context.lifecycle_stage != LLAMA_SELF_MODEL_EXTENSION_STAGE_TRANSIENT ||
        discovered_context.admission_score < 0.52f ||
        !found_discovered_scalar ||
        discovered_scalar.support_count == 0) {
        std::fprintf(stderr, "discovered-state lifecycle surfaces were not populated from primary feedback\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_model_extension_update tool_extension = llama_self_model_extension_default_update();
    tool_extension.source = LLAMA_SELF_MODEL_EXTENSION_SOURCE_TOOL_BASH_CLI;
    tool_extension.source_tool_kind = LLAMA_TOOL_KIND_BASH_CLI;
    tool_extension.kind = LLAMA_SELF_MODEL_EXTENSION_SCALAR_PARAM;
    tool_extension.domain = LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EPISTEMIC;
    tool_extension.flags =
            LLAMA_SELF_MODEL_EXTENSION_FLAG_ACTIVE |
            LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_GAIN |
            LLAMA_SELF_MODEL_EXTENSION_FLAG_HAS_DESIRED_STATE |
            LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS;
    tool_extension.value = 0.20f;
    tool_extension.desired_value = 1.0f;
    tool_extension.confidence = 0.80f;
    tool_extension.salience = 0.90f;
    tool_extension.gain_weight = 0.70f;
    tool_extension.allostatic_weight = 0.90f;
    std::snprintf(tool_extension.key, sizeof(tool_extension.key), "%s", "bash.answerability");
    std::snprintf(tool_extension.label, sizeof(tool_extension.label), "%s", "Bash Answerability");
    std::snprintf(tool_extension.content, sizeof(tool_extension.content), "%s", "CLI tool discovered whether the repository state is answerable.");

    if (llama_self_state_upsert_model_extension(ctx, tool_extension) != 0 ||
        llama_self_state_model_extension_count(ctx) != baseline_extension_count + 1) {
        std::fprintf(stderr, "failed to upsert tool-authored self-model extension\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_model_extension_info extension_info = {};
    if (!find_model_extension_by_key(ctx, "bash.answerability", &extension_info) ||
        extension_info.source_tool_kind != LLAMA_TOOL_KIND_BASH_CLI ||
        extension_info.kind != LLAMA_SELF_MODEL_EXTENSION_SCALAR_PARAM ||
        extension_info.domain != LLAMA_SELF_MODEL_EXTENSION_DOMAIN_EPISTEMIC ||
        extension_info.lifecycle_stage != LLAMA_SELF_MODEL_EXTENSION_STAGE_ALLOSTATIC ||
        extension_info.desired_value < 0.99f ||
        extension_info.value > 0.21f) {
        std::fprintf(stderr, "tool-authored self-model extension was not preserved\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_get_model_state(ctx, &model_state) != 0 ||
        model_state.extension_summary.active_count != baseline_extension_count + 1 ||
        model_state.extension_summary.gain_count < 1 ||
        model_state.extension_summary.allostatic_count < 1 ||
        model_state.extension_summary.allostatic_divergence <= 0.70f) {
        std::fprintf(stderr, "self-model extension summary did not reflect tool-authored allostatic state\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_remove_model_extension(ctx, "bash.answerability") != 0 ||
        llama_self_state_model_extension_count(ctx) != baseline_extension_count) {
        std::fprintf(stderr, "failed to remove tool-authored self-model extension\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_ANSWERABILITY, &register_info) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_USER_SATISFACTION_RISK, &register_info) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_RECOVERY_URGENCY, &register_info) != 0 ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_USER_DIRECTNESS_PREFERENCE, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_DISCOVERED_STATE_LOAD, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_DISCOVERED_STATE_PERMANENCE, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_DISCOVERED_STATE_ALLOSTATIC_LOAD, &register_info) != 0 ||
        register_info.scalar_value < 0.0f ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_USER_STRUCTURE_PREFERENCE, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f ||
        llama_self_state_get_register(ctx, LLAMA_SELF_REGISTER_USER_AUTONOMY_PREFERENCE, &register_info) != 0 ||
        register_info.scalar_value <= 0.0f) {
        std::fprintf(stderr, "expanded summary registers were not addressable\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    for (int i = 0; i < 3; ++i) {
        if (llama_self_state_note_validated_progress(ctx, 0.85f, 0.70f) != 0) {
            std::fprintf(stderr, "failed to apply validated progress for discovered-state consolidation\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }
    if (!find_model_extension_by_key(ctx, discovered_scalar.key, &extension_info) ||
        extension_info.lifecycle_stage != LLAMA_SELF_MODEL_EXTENSION_STAGE_ALLOSTATIC ||
        (extension_info.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS) == 0 ||
        (extension_info.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_HAS_DESIRED_STATE) == 0 ||
        extension_info.desired_value_max < extension_info.desired_value_min ||
        extension_info.allostatic_eligibility < 0.70f) {
        std::fprintf(stderr, "discovered scalar state did not consolidate into an allostatic objective\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_reactivation_info reactivation = {};
    if (llama_self_state_reactivation_count(ctx) <= 0 ||
        llama_self_state_get_reactivation(ctx, 0, &reactivation) != 0 ||
        reactivation.handle_id != 17 ||
        reactivation.priority <= 0.0f ||
        reactivation.top_similarity <= 0.0f) {
        std::fprintf(stderr, "reactivation priorities were not updated\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const int32_t system_turns_before_internal = social_state.system_turn_count;
    const std::vector<llama_token> internal_tokens = tokenize_or_die(vocab, "internal counterfactual planning artifact");
    llama_self_state_event internal_event = {
        /*.tokens =*/ internal_tokens.data(),
        /*.n_tokens =*/ internal_tokens.size(),
        /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
        /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL,
        /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_INTERNAL_ARTIFACT,
        /*.decoder_entropy =*/ 0.0f,
        /*.decoder_top_margin =*/ 1.0f,
        /*.artifact_kind =*/ LLAMA_SELF_COG_ARTIFACT_DMN_INTERNAL_WRITE,
        /*.loop_origin =*/ LLAMA_COG_COMMAND_ORIGIN_DMN,
        /*.phase =*/ LLAMA_COG_LOOP_PHASE_OBSERVE,
        /*.source_id =*/ 77,
        /*.plan_id =*/ 88,
    };
    llama_self_state_feature_vector internal_pre = {};
    llama_self_state_feature_vector internal_post = {};
    if (llama_self_state_build_prewrite_features(ctx, &internal_event, &internal_pre) != 0 ||
        llama_self_state_apply_prewrite(ctx, &internal_event, &internal_pre) != 0 ||
        llama_self_state_build_postwrite_features(ctx, &internal_event, &internal_post) != 0 ||
        llama_self_state_apply_postwrite(ctx, &internal_event, &internal_post) != 0 ||
        llama_self_state_get_social_state(ctx, &social_state) != 0 ||
        social_state.system_turn_count != system_turns_before_internal) {
        std::fprintf(stderr, "internal cognitive artifact mutated social turn counters\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_trace_item_info last_trace_item = {};
    const int32_t trace_index = llama_self_state_trace_count(ctx) - 1;
    if (llama_self_state_trace_token_count(ctx) <= 0 ||
        trace_index < 0 ||
        llama_self_state_trace_get_item(ctx, trace_index, &last_trace_item) != 0 ||
        last_trace_item.artifact_kind != LLAMA_SELF_COG_ARTIFACT_DMN_INTERNAL_WRITE ||
        last_trace_item.loop_origin != LLAMA_COG_COMMAND_ORIGIN_DMN ||
        last_trace_item.phase != LLAMA_COG_LOOP_PHASE_OBSERVE ||
        last_trace_item.source_id != 77 ||
        last_trace_item.plan_id != 88 ||
        last_trace_item.token_count != (int32_t) internal_tokens.size()) {
        std::fprintf(stderr, "internal cognitive artifact trace metadata was not preserved\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const int32_t expected_trace_count = llama_self_state_trace_count(ctx);
    const int32_t expected_working_memory_count = llama_self_state_working_memory_count(ctx);
    if (expected_trace_count <= 0 ||
        expected_working_memory_count <= 0 ||
        llama_self_state_replay_trace(ctx, 1) != 0 ||
        llama_self_state_working_memory_count(ctx) != 1 ||
        llama_self_state_trace_count(ctx) != expected_trace_count ||
        llama_self_state_replay_trace(ctx, -1) != 0 ||
        llama_self_state_working_memory_count(ctx) != expected_working_memory_count ||
        llama_self_state_trace_count(ctx) != expected_trace_count) {
        std::fprintf(stderr, "trace replay did not preserve deterministic state\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_get_model_state(ctx, &model_state) != 0 ||
        !model_state.prediction_error.valid ||
        model_state.prediction_error.steps_error < 0.0f ||
        model_state.prediction_error.goal_progress_error < 0.0f ||
        !model_state.belief_summary.valid ||
        model_state.belief_summary.residual_allostatic_pressure <= 0.0f ||
        model_state.belief_summary.max_slot_pressure <= 0.0f) {
        std::fprintf(stderr, "self-model forecast error trace was not preserved\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_state_params no_belief = llama_self_state_default_params();
    no_belief.enable_belief_state = false;
    no_belief.belief_slot_count = 0;
    if (llama_self_state_configure(ctx, no_belief) != 0 ||
        llama_self_state_get_model_state(ctx, &model_state) != 0 ||
        model_state.belief_summary.valid ||
        model_state.belief_slot_count != 0 ||
        model_state.promotion_candidate_count != 0) {
        std::fprintf(stderr, "belief-state disable path did not clear bounded uncertainty surface\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (llama_self_state_configure(ctx, llama_self_state_default_params()) != 0) {
        std::fprintf(stderr, "failed to restore belief-state defaults\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_updater_program program = llama_self_state_default_updater_program();
    program.version = 2;
    program.broadcast_social_weight = 0.05f;
    program.broadcast_contradiction_weight = 0.45f;
    program.broadcast_goal_weight = 0.25f;
    program.repair_emit_threshold = 0.58f;
    program.repair_dissatisfaction_floor = 0.32f;
    program.repair_recent_user_valence_floor = 0.18f;
    program.repair_inhibition_max = 0.74f;
    program.repair_admission_floor = 0.28f;
    program.repair_admission_weight = 0.41f;
    if (program.rule_count == 0) {
        std::fprintf(stderr, "default updater program is missing register rules\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    for (uint32_t i = 0; i < program.rule_count; ++i) {
        if (program.rules[i].register_id == LLAMA_SELF_REGISTER_BROADCAST_PRESSURE &&
            program.rules[i].phase_mask == LLAMA_SELF_UPDATER_PHASE_POSTWRITE) {
            program.rules[i].rise_gain = 0.75f;
            program.rules[i].feature_weights[0] = 0.82f;
        }
    }

    if (llama_self_state_set_updater_program(ctx, program) != 0) {
        std::fprintf(stderr, "failed to set updater program\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_updater_program invalid_program = program;
    invalid_program.version = 3;
    invalid_program.rules[0].register_id = LLAMA_SELF_REGISTER_CHANNEL_STATE;
    if (llama_self_state_set_updater_program(ctx, invalid_program) == 0) {
        std::fprintf(stderr, "accepted invalid updater program rule\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_self_updater_program program_roundtrip = {};
    llama_self_counterfactual_result counterfactual = {};
    if (llama_self_state_get_updater_program(ctx, &program_roundtrip) != 0 ||
        program_roundtrip.version != 2 ||
        program_roundtrip.rule_count == 0 ||
        program_roundtrip.rules[0].phase_mask == 0 ||
        program_roundtrip.repair_emit_threshold != program.repair_emit_threshold ||
        program_roundtrip.repair_dissatisfaction_floor != program.repair_dissatisfaction_floor ||
        program_roundtrip.repair_recent_user_valence_floor != program.repair_recent_user_valence_floor ||
        program_roundtrip.repair_inhibition_max != program.repair_inhibition_max ||
        program_roundtrip.repair_admission_floor != program.repair_admission_floor ||
        program_roundtrip.repair_admission_weight != program.repair_admission_weight ||
        llama_self_state_evaluate_counterfactual(ctx, program, -1, &counterfactual) != 0 ||
        counterfactual.updater_version != 2 ||
        counterfactual.replay_channel != LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL ||
        counterfactual.replayed_events != expected_trace_count ||
        counterfactual.working_memory_count != expected_working_memory_count ||
        counterfactual.broadcast_pressure <= 0.0f) {
        std::fprintf(stderr, "counterfactual updater evaluation failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const size_t trace_size = llama_self_state_trace_export_size(ctx);
    std::vector<uint8_t> trace_blob(trace_size);
    if (trace_size == 0 ||
        llama_self_state_trace_export(ctx, trace_blob.data(), trace_blob.size()) != 0) {
        std::fprintf(stderr, "failed to export self-state trace\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_context * counterfactual_ctx = llama_init_from_model(model, cparams);
    if (!counterfactual_ctx) {
        std::fprintf(stderr, "failed to create counterfactual context\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    if (llama_self_state_trace_import(counterfactual_ctx, trace_blob.data(), trace_blob.size(), true) != 0 ||
        llama_self_state_replay_trace_on_channel(counterfactual_ctx, -1, LLAMA_SELF_STATE_EVENT_CHANNEL_COUNTERFACTUAL) != 0 ||
        llama_self_state_get_social_state(counterfactual_ctx, &social_state) != 0 ||
        social_state.user_turn_count != 0 ||
        social_state.system_turn_count != 0 ||
        llama_self_state_get_register(counterfactual_ctx, LLAMA_SELF_REGISTER_CHANNEL_STATE, &register_info) != 0 ||
        register_info.categorical_value != LLAMA_SELF_STATE_CHANNEL_WAITING) {
        std::fprintf(stderr, "counterfactual replay leaked into primary-channel state\n");
        llama_free(counterfactual_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    llama_free(counterfactual_ctx);

    llama_context * imported_ctx = llama_init_from_model(model, cparams);
    if (!imported_ctx) {
        std::fprintf(stderr, "failed to create imported context\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_self_state_trace_import(imported_ctx, trace_blob.data(), trace_blob.size(), true) != 0 ||
        llama_self_state_trace_count(imported_ctx) != expected_trace_count ||
        llama_self_state_trace_get_item(imported_ctx, expected_trace_count - 1, &last_trace_item) != 0 ||
        last_trace_item.artifact_kind != LLAMA_SELF_COG_ARTIFACT_DMN_INTERNAL_WRITE ||
        last_trace_item.plan_id != 88 ||
        llama_self_state_replay_trace(imported_ctx, -1) != 0 ||
        llama_self_state_working_memory_count(imported_ctx) != expected_working_memory_count) {
        std::fprintf(stderr, "trace import/replay failed\n");
        llama_free(imported_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(imported_ctx);

    {
        llama_context * compaction_ctx = llama_init_from_model(model, cparams);
        if (!compaction_ctx) {
            std::fprintf(stderr, "failed to create compaction context\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const llama_self_state_time_point compaction_time = {
            /*.wall_clock_ms =*/ 1700000005000,
            /*.monotonic_ms =*/ 5000,
            /*.timezone_offset_minutes =*/ -360,
        };
        if (llama_self_state_set_time(compaction_ctx, compaction_time) != 0) {
            std::fprintf(stderr, "failed to seed compaction context time\n");
            llama_free(compaction_ctx);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const int iteration_count = 160;
        for (int i = 0; i < iteration_count; ++i) {
            std::vector<llama_token> tokens(48);
            for (size_t token_idx = 0; token_idx < tokens.size(); ++token_idx) {
                tokens[token_idx] = (llama_token) (100 + ((i * 53 + (int) token_idx) % 997));
            }
            llama_self_state_event artifact_event = {
                /*.tokens =*/ tokens.data(),
                /*.n_tokens =*/ tokens.size(),
                /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
                /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
                /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_INTERNAL_ARTIFACT,
                /*.decoder_entropy =*/ 0.0f,
                /*.decoder_top_margin =*/ 1.0f,
                /*.artifact_kind =*/ LLAMA_SELF_COG_ARTIFACT_ACTIVE_PLAN,
                /*.loop_origin =*/ LLAMA_COG_COMMAND_ORIGIN_ACTIVE,
                /*.phase =*/ LLAMA_FUNCTIONAL_MICROPHASE_PLAN_COMPOSE,
                /*.source_id =*/ i,
                /*.plan_id =*/ i,
            };
            llama_self_state_feature_vector pre = {};
            llama_self_state_feature_vector post = {};
            if (llama_self_state_build_prewrite_features(compaction_ctx, &artifact_event, &pre) != 0 ||
                llama_self_state_apply_prewrite(compaction_ctx, &artifact_event, &pre) != 0 ||
                llama_self_state_build_postwrite_features(compaction_ctx, &artifact_event, &post) != 0 ||
                llama_self_state_apply_postwrite(compaction_ctx, &artifact_event, &post) != 0) {
                std::fprintf(stderr, "failed to append compaction artifact\n");
                llama_free(compaction_ctx);
                llama_free(ctx);
                llama_model_free(model);
                return 1;
            }
        }

        llama_self_trace_item_info first_compacted_item = {};
        if (llama_self_state_trace_count(compaction_ctx) >= iteration_count ||
            llama_self_state_trace_token_count(compaction_ctx) > 2048 ||
            llama_self_state_trace_get_item(compaction_ctx, 0, &first_compacted_item) != 0 ||
            first_compacted_item.source_id <= 0) {
            std::fprintf(stderr, "shared cognitive trace compaction did not evict oldest artifacts first\n");
            llama_free(compaction_ctx);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(compaction_ctx);
    }

    if (!verify_runtime_persistence_surfaces(ctx, model, cparams, program, expected_trace_count)) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    {
        mock_supermemory_server mock_server;
        if (!mock_server.start()) {
            std::fprintf(stderr, "failed to start mock hard-memory server\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_context * hard_memory_ctx = llama_init_from_model(model, cparams);
        if (!hard_memory_ctx) {
            std::fprintf(stderr, "failed to create hard-memory context\n");
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_hard_memory_config hard_memory = llama_hard_memory_default_config();
        hard_memory.enabled = true;
        hard_memory.archive_enabled = true;
        hard_memory.include_profile_by_default = true;
        hard_memory.max_results = 2;
        hard_memory.query_threshold = 0.33f;
        hard_memory.archival_delta_threshold = 0.05f;
        std::snprintf(hard_memory.base_url, sizeof(hard_memory.base_url), "http://127.0.0.1:%d", mock_server.port);
        std::snprintf(hard_memory.auth_token, sizeof(hard_memory.auth_token), "%s", "sm_test_token");
        std::snprintf(hard_memory.container_tag, sizeof(hard_memory.container_tag), "%s", "vicuna-self-state");
        std::snprintf(hard_memory.runtime_identity, sizeof(hard_memory.runtime_identity), "%s", "vicuna-tests");

        if (llama_hard_memory_configure(hard_memory_ctx, hard_memory) != 0) {
            std::fprintf(stderr, "failed to configure hard memory\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_hard_memory_query_request hard_memory_query_req = {};
        hard_memory_query_req.include_profile = true;
        hard_memory_query_req.limit = 2;
        hard_memory_query_req.threshold = 0.40f;
        std::snprintf(hard_memory_query_req.query, sizeof(hard_memory_query_req.query), "%s", "strict memory runtime");

        llama_hard_memory_result hard_memory_result = {};
        if (llama_hard_memory_query(hard_memory_ctx, &hard_memory_query_req, &hard_memory_result) != 0 ||
            !hard_memory_result.ok ||
            hard_memory_result.result_count != 2 ||
            std::string(hard_memory_result.results[0].id).empty() ||
            std::string(hard_memory_result.profile_static).find("explicit runtime policy") == std::string::npos ||
            std::string(hard_memory_result.effective_container_tag) != "vicuna-self-state" ||
            hard_memory_result.tool_kind != LLAMA_TOOL_KIND_HARD_MEMORY_QUERY) {
            std::fprintf(stderr, "hard-memory query did not return bounded typed results\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (hard_memory_result.results[0].kind != LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT ||
            hard_memory_result.results[0].domain != LLAMA_HARD_MEMORY_DOMAIN_EPISTEMIC ||
            hard_memory_result.results[0].source_tool_kind != LLAMA_TOOL_KIND_HARD_MEMORY_QUERY ||
            hard_memory_result.results[0].gain_bias < 0.87f ||
            std::string(hard_memory_result.results[0].tags[1]) != "self_model" ||
            hard_memory_result.retrieval_summary.self_model_count != 1 ||
            hard_memory_result.retrieval_summary.user_model_count != 1 ||
            hard_memory_result.retrieval_summary.mean_similarity < 0.86f ||
            hard_memory_result.retrieval_summary.max_similarity < 0.90f ||
            hard_memory_result.retrieval_summary.gain_support <= 0.60f ||
            hard_memory_result.retrieval_summary.epistemic_support <= 0.0f ||
            hard_memory_result.retrieval_summary.user_support <= 0.0f) {
            std::fprintf(stderr, "hard-memory query metadata was not parsed into typed retrieval summaries\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_self_model_state_info hard_memory_model_state = {};
        llama_self_model_extension_info hard_memory_extension = {};
        if (llama_self_state_model_extension_count(hard_memory_ctx) <= 0 ||
            llama_self_state_get_model_state(hard_memory_ctx, &hard_memory_model_state) != 0 ||
            !hard_memory_model_state.last_extension_trace.valid ||
            hard_memory_model_state.last_extension_trace.promoted_count <= 0 ||
            hard_memory_model_state.extension_summary.hard_memory_count <= 0 ||
            hard_memory_model_state.extension_summary.discovered_count <= 0 ||
            hard_memory_model_state.extension_summary.allostatic_count != 0 ||
            hard_memory_model_state.extension_summary.context_activation <= 0.0f ||
            llama_self_state_get_model_extension(hard_memory_ctx, 0, &hard_memory_extension) != 0 ||
            hard_memory_extension.source_tool_kind != LLAMA_TOOL_KIND_HARD_MEMORY_QUERY ||
            hard_memory_extension.lifecycle_stage != LLAMA_SELF_MODEL_EXTENSION_STAGE_TRANSIENT ||
            (hard_memory_extension.flags & LLAMA_SELF_MODEL_EXTENSION_FLAG_AFFECT_ALLOSTASIS) != 0) {
            std::fprintf(stderr, "hard-memory query did not promote bounded non-allostatic self-model extensions\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_hard_memory_result last_result = {};
        if (llama_hard_memory_get_last_result(hard_memory_ctx, &last_result) != 0 ||
            last_result.result_count != hard_memory_result.result_count) {
            std::fprintf(stderr, "hard-memory last-result surface was not preserved\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> delta_tokens = tokenize_or_die(vocab, "This was wrong, frustrating, and probably contradicted the earlier plan.");
        llama_self_state_event delta_event = {
            /*.tokens =*/ delta_tokens.data(),
            /*.n_tokens =*/ delta_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
            /*.decoder_entropy =*/ 1.5f,
            /*.decoder_top_margin =*/ 0.05f,
        };
        llama_self_state_feature_vector delta_pre = {};
        llama_self_state_feature_vector delta_post = {};
        if (llama_self_state_build_prewrite_features(hard_memory_ctx, &delta_event, &delta_pre) != 0 ||
            llama_self_state_apply_prewrite(hard_memory_ctx, &delta_event, &delta_pre) != 0 ||
            llama_self_state_build_postwrite_features(hard_memory_ctx, &delta_event, &delta_post) != 0 ||
            llama_self_state_apply_postwrite(hard_memory_ctx, &delta_event, &delta_post) != 0) {
            std::fprintf(stderr, "failed to drive hard-memory archival event\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_hard_memory_archive_trace archive_trace = {};
        if (llama_hard_memory_get_last_archive_trace(hard_memory_ctx, &archive_trace) != 0 ||
            !archive_trace.archived ||
            archive_trace.delta.total_delta < hard_memory.archival_delta_threshold ||
            archive_trace.delta.dimension_count <= 0 ||
            std::string(archive_trace.container_tag) != "vicuna-self-state" ||
            archive_trace.primitive_count < 2 ||
            std::string(archive_trace.content_excerpt).empty() ||
            mock_server.archive_calls.load() != 1) {
            std::fprintf(stderr, "hard-memory archival did not persist above-threshold delta context\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        bool saw_event_primitive = false;
        bool saw_user_or_self_model_primitive = false;
        for (int32_t i = 0; i < archive_trace.primitive_count; ++i) {
            saw_event_primitive = saw_event_primitive ||
                    archive_trace.primitives[i].kind == LLAMA_HARD_MEMORY_PRIMITIVE_EVENT_FRAGMENT;
            saw_user_or_self_model_primitive = saw_user_or_self_model_primitive ||
                    archive_trace.primitives[i].kind == LLAMA_HARD_MEMORY_PRIMITIVE_USER_MODEL ||
                    archive_trace.primitives[i].kind == LLAMA_HARD_MEMORY_PRIMITIVE_SELF_MODEL_FRAGMENT;
        }
        if (!saw_event_primitive || !saw_user_or_self_model_primitive) {
            std::fprintf(stderr, "hard-memory archival did not preserve typed primitive mix\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::string archive_body = [&]() {
            std::lock_guard<std::mutex> lock(mock_server.mutex);
            return mock_server.last_archive_body;
        }();
        if (archive_body.find("\"runtimeIdentity\":\"vicuna-tests\"") == std::string::npos ||
            archive_body.find("\"containerTag\":\"vicuna-self-state\"") == std::string::npos) {
            std::fprintf(stderr, "hard-memory archival did not preserve self-host routing metadata\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const json archive_payload = json::parse(archive_body);
        if (!archive_payload.contains("memories") ||
            !archive_payload["memories"].is_array() ||
            archive_payload["memories"].size() < 2) {
            std::fprintf(stderr, "hard-memory archival payload did not preserve primitive batch structure\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        bool saw_archived_event = false;
        bool saw_archived_model = false;
        bool saw_changed_registers = false;
        bool saw_preference_tag = false;
        bool saw_rhetoric_tag = false;
        for (const auto & item : archive_payload["memories"]) {
            if (!item.is_object()) {
                continue;
            }
            const json metadata = item.value("metadata", json::object());
            const std::string kind = metadata.value("kind", std::string());
            saw_archived_event = saw_archived_event ||
                    kind == "event_fragment";
            saw_archived_model = saw_archived_model ||
                    kind == "user_model" || kind == "self_model_fragment";
            saw_changed_registers = saw_changed_registers ||
                    (metadata.contains("changedRegisters") && metadata["changedRegisters"].is_array() && !metadata["changedRegisters"].empty());
            if (kind == "user_model" && metadata.contains("tags") && metadata["tags"].is_array()) {
                for (const auto & tag : metadata["tags"]) {
                    if (!tag.is_string()) {
                        continue;
                    }
                    saw_preference_tag = saw_preference_tag || tag.get<std::string>() == "preference";
                    saw_rhetoric_tag = saw_rhetoric_tag || tag.get<std::string>() == "rhetoric";
                }
            }
        }
        if (!saw_archived_event || !saw_archived_model || !saw_changed_registers ||
            !saw_preference_tag || !saw_rhetoric_tag) {
            std::fprintf(stderr, "hard-memory archival payload did not include typed metadata-rich primitives\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        hard_memory.archival_delta_threshold = 10.0f;
        if (llama_hard_memory_configure(hard_memory_ctx, hard_memory) != 0) {
            std::fprintf(stderr, "failed to raise hard-memory archival threshold\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        const std::vector<llama_token> mild_tokens = tokenize_or_die(vocab, "A mild follow-up.");
        llama_self_state_event mild_event = {
            /*.tokens =*/ mild_tokens.data(),
            /*.n_tokens =*/ mild_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
            /*.decoder_entropy =*/ 0.1f,
            /*.decoder_top_margin =*/ 0.9f,
        };
        if (llama_self_state_build_prewrite_features(hard_memory_ctx, &mild_event, &delta_pre) != 0 ||
            llama_self_state_apply_prewrite(hard_memory_ctx, &mild_event, &delta_pre) != 0 ||
            llama_self_state_build_postwrite_features(hard_memory_ctx, &mild_event, &delta_post) != 0 ||
            llama_self_state_apply_postwrite(hard_memory_ctx, &mild_event, &delta_post) != 0 ||
            mock_server.archive_calls.load() != 1) {
            std::fprintf(stderr, "hard-memory archival threshold did not suppress low-perturbation writes\n");
            llama_free(hard_memory_ctx);
            mock_server.stop();
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_free(hard_memory_ctx);
        mock_server.stop();
    }

    {
        const int32_t base_trace_count = llama_self_state_trace_count(ctx);
        const int32_t base_working_memory = llama_self_state_working_memory_count(ctx);
        const std::vector<llama_token> user_tokens = tokenize_or_die(vocab, "While you were away I asked for a status update.");
        const std::vector<llama_token> system_tokens = tokenize_or_die(vocab, "I reviewed the build logs and queued the next repair step.");
        const std::vector<llama_token> tool_tokens = tokenize_or_die(vocab, "bash tool completed: git status showed one modified runtime file.");

        auto admit_event = [ctx](const llama_self_state_event & event) {
            llama_self_state_feature_vector pre = {};
            llama_self_state_feature_vector post = {};
            return llama_self_state_build_prewrite_features(ctx, &event, &pre) == 0 &&
                    llama_self_state_apply_prewrite(ctx, &event, &pre) == 0 &&
                    llama_self_state_build_postwrite_features(ctx, &event, &post) == 0 &&
                    llama_self_state_apply_postwrite(ctx, &event, &post) == 0;
        };

        const llama_self_state_event user_event = {
            /*.tokens =*/ user_tokens.data(),
            /*.n_tokens =*/ user_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_USER,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        const llama_self_state_event system_event = {
            /*.tokens =*/ system_tokens.data(),
            /*.n_tokens =*/ system_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_SYSTEM,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_EMIT_FOLLOWUP,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };
        const llama_self_state_event tool_event = {
            /*.tokens =*/ tool_tokens.data(),
            /*.n_tokens =*/ tool_tokens.size(),
            /*.role =*/ LLAMA_SELF_STATE_EVENT_TOOL,
            /*.channel =*/ LLAMA_SELF_STATE_EVENT_CHANNEL_PRIMARY,
            /*.flags =*/ LLAMA_SELF_STATE_EVENT_ADMITTED | LLAMA_SELF_STATE_EVENT_TOOL_COMPLETED,
            /*.decoder_entropy =*/ 0.0f,
            /*.decoder_top_margin =*/ 1.0f,
        };

        if (!admit_event(user_event) ||
            !admit_event(system_event) ||
            !admit_event(tool_event)) {
            std::fprintf(stderr, "failed to admit shared conversation/tool/system events\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (llama_self_state_trace_count(ctx) < base_trace_count + 3 ||
            llama_self_state_working_memory_count(ctx) < base_working_memory + 3) {
            std::fprintf(stderr, "shared context events did not enter the unified trace and working memory\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        llama_self_trace_item_info trace_item = {};
        if (llama_self_state_trace_get_item(ctx, llama_self_state_trace_count(ctx) - 3, &trace_item) != 0 ||
            trace_item.role != LLAMA_SELF_STATE_EVENT_USER ||
            llama_self_state_trace_get_item(ctx, llama_self_state_trace_count(ctx) - 2, &trace_item) != 0 ||
            trace_item.role != LLAMA_SELF_STATE_EVENT_SYSTEM ||
            llama_self_state_trace_get_item(ctx, llama_self_state_trace_count(ctx) - 1, &trace_item) != 0 ||
            trace_item.role != LLAMA_SELF_STATE_EVENT_TOOL) {
            std::fprintf(stderr, "shared context events were not preserved in one ordered trace\n");
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    if (llama_self_state_clear_trace(ctx) != 0 ||
        llama_self_state_trace_count(ctx) != 0) {
        std::fprintf(stderr, "trace clear failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
