#include "server-decode-controller.h"

#include <cmath>
#include <fstream>

namespace {

static float clamp_nonzero_std(float value) {
    return std::fabs(value) < 1e-6f ? 1.0f : value;
}

static float sigmoid_local(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

static bool parse_float_vector(const json & payload, std::vector<float> * out_vector) {
    if (!payload.is_array() || out_vector == nullptr) {
        return false;
    }
    out_vector->clear();
    out_vector->reserve(payload.size());
    for (const auto & item : payload) {
        if (!item.is_number()) {
            return false;
        }
        out_vector->push_back(item.get<float>());
    }
    return true;
}

static bool parse_float_matrix(
        const json & payload,
        size_t expected_rows,
        size_t expected_cols,
        std::vector<std::vector<float>> * out_matrix) {
    if (!payload.is_array() || out_matrix == nullptr || payload.size() != expected_rows) {
        return false;
    }
    out_matrix->clear();
    out_matrix->reserve(expected_rows);
    for (const auto & row_json : payload) {
        std::vector<float> row;
        if (!parse_float_vector(row_json, &row) || row.size() != expected_cols) {
            return false;
        }
        out_matrix->push_back(std::move(row));
    }
    return true;
}

static void append_decode_policy_features(
        const server_decode_policy_config & decode_policy,
        std::vector<float> * out_features) {
    if (out_features == nullptr) {
        return;
    }
    out_features->insert(out_features->end(), {
        decode_policy.base_temperature,
        static_cast<float>(decode_policy.base_top_k),
        decode_policy.base_top_p,
        decode_policy.base_min_p,
        decode_policy.control_limits.min_temperature,
        decode_policy.control_limits.max_temperature,
        decode_policy.control_limits.min_top_p,
        decode_policy.control_limits.max_top_p,
        decode_policy.control_limits.min_min_p,
        decode_policy.control_limits.max_min_p,
        decode_policy.control_limits.min_typical_p,
        decode_policy.control_limits.max_typical_p,
        decode_policy.control_limits.max_top_n_sigma,
        decode_policy.control_limits.min_repeat_penalty,
        decode_policy.control_limits.max_repeat_penalty,
        decode_policy.control_limits.max_frequency_penalty,
        decode_policy.control_limits.max_presence_penalty,
        decode_policy.control_limits.max_dry_multiplier,
        static_cast<float>(decode_policy.control_limits.max_penalty_last_n),
        static_cast<float>(decode_policy.control_limits.max_dry_allowed_length),
        static_cast<float>(decode_policy.control_limits.max_dry_penalty_last_n),
    });
}

static void append_mask_features(
        const llama_runtime_control_action_mask & mask,
        std::vector<float> * out_features) {
    if (out_features == nullptr) {
        return;
    }
    out_features->insert(out_features->end(), {
        mask.allow_sampling ? 1.0f : 0.0f,
        mask.allow_repetition ? 1.0f : 0.0f,
        mask.allow_structure ? 1.0f : 0.0f,
        mask.allow_branch ? 1.0f : 0.0f,
        mask.allow_steering ? 1.0f : 0.0f,
        static_cast<float>(mask.max_branch_sample_count),
    });
}

static std::vector<float> flatten_observation_features(
        const server_decode_control_observation & observation,
        bool include_decode_policy) {
    const auto & moment = observation.moment;
    const auto & vad = observation.vad;
    const auto & runtime = observation.runtime_signals;
    const auto & bundles = observation.bundles;
    const auto & previous = observation.previous_executed_action;
    std::vector<float> features = {
        moment.confidence,
        moment.curiosity,
        moment.frustration,
        moment.satisfaction,
        moment.momentum,
        moment.caution,
        moment.stall,
        moment.epistemic_pressure,
        moment.planning_clarity,
        moment.user_alignment,
        moment.semantic_novelty,
        moment.runtime_trust,
        moment.runtime_failure_pressure,
        moment.contradiction_pressure,
        vad.valence,
        vad.arousal,
        vad.dominance,
        runtime.mean_entropy,
        runtime.max_entropy,
        runtime.mean_margin,
        runtime.sampled_prob,
        runtime.stop_prob,
        runtime.repeat_hit_rate,
        runtime.route_entropy_mean,
        runtime.route_entropy_max,
        runtime.route_top1_weight_mean,
        runtime.route_top1_weight_max,
        runtime.attention_entropy_mean,
        runtime.attention_entropy_max,
        runtime.attention_top1_mass_mean,
        runtime.attention_top1_mass_max,
        runtime.agreement_score,
        runtime.consistency_entropy,
        runtime.branch_disagreement,
        runtime.verifier_disagreement,
        runtime.graph_value_mean_abs,
        runtime.graph_value_rms,
        runtime.graph_value_max_abs,
        runtime.dominant_expert_fraction_top1,
        runtime.dominant_expert_fraction_mass,
        runtime.timing_decode_ms,
        runtime.timing_sample_ms,
        runtime.timing_delta_ms,
        runtime.memory_budget_ratio,
        runtime.attention_budget_ratio,
        runtime.recurrent_budget_ratio,
        static_cast<float>(runtime.candidate_count),
        static_cast<float>(runtime.attention_pos_min),
        static_cast<float>(runtime.attention_pos_max),
        static_cast<float>(runtime.recurrent_pos_min),
        static_cast<float>(runtime.recurrent_pos_max),
        static_cast<float>(runtime.expert_count),
        static_cast<float>(runtime.experts_selected),
        static_cast<float>(runtime.dominant_expert_count),
        static_cast<float>(runtime.comparison_count),
        static_cast<float>(runtime.semantic_group_count),
        static_cast<float>(runtime.status_code),
        runtime.runtime_failure ? 1.0f : 0.0f,
        runtime.verifier_active ? 1.0f : 0.0f,
        runtime.grammar_active ? 1.0f : 0.0f,
        runtime.logit_bias_active ? 1.0f : 0.0f,
        runtime.backend_sampler ? 1.0f : 0.0f,
        runtime.optimized ? 1.0f : 0.0f,
        runtime.prompt_section_changed ? 1.0f : 0.0f,
        bundles.uncertainty_regulation,
        bundles.anti_repetition_recovery,
        bundles.structural_validity,
        bundles.verification_pressure,
        bundles.commit_efficiency,
        bundles.steering_pressure,
    };
    if (include_decode_policy) {
        append_decode_policy_features(observation.decode_policy, &features);
        append_mask_features(observation.decode_policy.action_mask, &features);
    } else {
        append_mask_features(observation.mask, &features);
    }
    features.insert(features.end(), {
        observation.previous_executed_action_available ? 1.0f : 0.0f,
        previous.valid ? 1.0f : 0.0f,
        previous.sampling.enabled ? 1.0f : 0.0f,
        previous.sampling.has_temperature ? 1.0f : 0.0f,
        previous.sampling.has_top_k ? 1.0f : 0.0f,
        previous.sampling.has_top_p ? 1.0f : 0.0f,
        previous.sampling.has_min_p ? 1.0f : 0.0f,
        previous.sampling.has_typical_p ? 1.0f : 0.0f,
        previous.sampling.has_top_n_sigma ? 1.0f : 0.0f,
        previous.repetition.enabled ? 1.0f : 0.0f,
        previous.repetition.has_repeat_penalty ? 1.0f : 0.0f,
        previous.repetition.has_frequency_penalty ? 1.0f : 0.0f,
        previous.repetition.has_presence_penalty ? 1.0f : 0.0f,
        previous.repetition.has_penalty_last_n ? 1.0f : 0.0f,
        previous.structure.enabled ? 1.0f : 0.0f,
        previous.structure.clear_grammar ? 1.0f : 0.0f,
        previous.structure.clear_logit_bias ? 1.0f : 0.0f,
        previous.branch.enabled ? 1.0f : 0.0f,
        previous.branch.checkpoint_now ? 1.0f : 0.0f,
        previous.branch.request_verify ? 1.0f : 0.0f,
        previous.steering.enabled ? 1.0f : 0.0f,
        previous.steering.clear_cvec ? 1.0f : 0.0f,
        previous.sampling.temperature,
        static_cast<float>(previous.sampling.top_k),
        previous.sampling.top_p,
        previous.sampling.min_p,
        previous.sampling.typical_p,
        previous.sampling.top_n_sigma,
        static_cast<float>(previous.sampling.min_keep),
        previous.repetition.repeat_penalty,
        previous.repetition.frequency_penalty,
        previous.repetition.presence_penalty,
        static_cast<float>(previous.repetition.penalty_last_n),
        static_cast<float>(previous.branch.checkpoint_slot),
        static_cast<float>(previous.branch.restore_slot),
        static_cast<float>(previous.branch.branch_sample_count),
    });
    return features;
}

static std::vector<float> linear_forward(
        const server_decode_controller_linear_head & head,
        const std::vector<float> & input) {
    std::vector<float> out(head.bias.size(), 0.0f);
    for (size_t row = 0; row < head.weights.size(); ++row) {
        float total = head.bias[row];
        for (size_t col = 0; col < head.weights[row].size(); ++col) {
            total += head.weights[row][col] * input[col];
        }
        out[row] = total;
    }
    return out;
}

static void apply_action_boolean(llama_runtime_control_action * action, const std::string & name, bool value) {
    if (name == "valid") {
        action->valid = value;
    } else if (name == "sampling.enabled") {
        action->sampling.enabled = value;
    } else if (name == "sampling.has_temperature") {
        action->sampling.has_temperature = value;
    } else if (name == "sampling.has_top_k") {
        action->sampling.has_top_k = value;
    } else if (name == "sampling.has_top_p") {
        action->sampling.has_top_p = value;
    } else if (name == "sampling.has_min_p") {
        action->sampling.has_min_p = value;
    } else if (name == "sampling.has_typical_p") {
        action->sampling.has_typical_p = value;
    } else if (name == "sampling.has_top_n_sigma") {
        action->sampling.has_top_n_sigma = value;
    } else if (name == "repetition.enabled") {
        action->repetition.enabled = value;
    } else if (name == "repetition.has_repeat_penalty") {
        action->repetition.has_repeat_penalty = value;
    } else if (name == "repetition.has_frequency_penalty") {
        action->repetition.has_frequency_penalty = value;
    } else if (name == "repetition.has_presence_penalty") {
        action->repetition.has_presence_penalty = value;
    } else if (name == "repetition.has_penalty_last_n") {
        action->repetition.has_penalty_last_n = value;
    } else if (name == "structure.enabled") {
        action->structure.enabled = value;
    } else if (name == "structure.clear_grammar") {
        action->structure.clear_grammar = value;
    } else if (name == "structure.clear_logit_bias") {
        action->structure.clear_logit_bias = value;
    } else if (name == "branch.enabled") {
        action->branch.enabled = value;
    } else if (name == "branch.checkpoint_now") {
        action->branch.checkpoint_now = value;
    } else if (name == "branch.request_verify") {
        action->branch.request_verify = value;
    } else if (name == "steering.enabled") {
        action->steering.enabled = value;
    } else if (name == "steering.clear_cvec") {
        action->steering.clear_cvec = value;
    }
}

static void apply_action_numeric(llama_runtime_control_action * action, const std::string & name, float value) {
    if (name == "sampling.temperature") {
        action->sampling.temperature = value;
    } else if (name == "sampling.top_k") {
        action->sampling.top_k = static_cast<int32_t>(std::round(value));
    } else if (name == "sampling.top_p") {
        action->sampling.top_p = value;
    } else if (name == "sampling.min_p") {
        action->sampling.min_p = value;
    } else if (name == "sampling.typical_p") {
        action->sampling.typical_p = value;
    } else if (name == "sampling.top_n_sigma") {
        action->sampling.top_n_sigma = value;
    } else if (name == "sampling.min_keep") {
        action->sampling.min_keep = static_cast<int32_t>(std::round(value));
    } else if (name == "repetition.repeat_penalty") {
        action->repetition.repeat_penalty = value;
    } else if (name == "repetition.frequency_penalty") {
        action->repetition.frequency_penalty = value;
    } else if (name == "repetition.presence_penalty") {
        action->repetition.presence_penalty = value;
    } else if (name == "repetition.penalty_last_n") {
        action->repetition.penalty_last_n = static_cast<int32_t>(std::round(value));
    } else if (name == "branch.checkpoint_slot") {
        action->branch.checkpoint_slot = static_cast<int32_t>(std::round(value));
    } else if (name == "branch.restore_slot") {
        action->branch.restore_slot = static_cast<int32_t>(std::round(value));
    } else if (name == "branch.branch_sample_count") {
        action->branch.branch_sample_count = static_cast<int32_t>(std::round(value));
    }
}

static std::string argmax_vocab(const std::vector<float> & logits, const std::vector<std::string> & vocab, float * out_confidence) {
    if (logits.empty() || vocab.empty()) {
        if (out_confidence != nullptr) {
            *out_confidence = 0.0f;
        }
        return "";
    }
    size_t best_index = 0;
    float best_value = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best_index = i;
        }
    }
    double sum = 0.0;
    std::vector<double> exps(logits.size(), 0.0);
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(static_cast<double>(logits[i] - best_value));
        sum += exps[i];
    }
    if (out_confidence != nullptr) {
        *out_confidence = static_cast<float>(exps[best_index] / std::max(sum, 1e-12));
    }
    return best_index < vocab.size() ? vocab[best_index] : std::string();
}

static float clamp_numeric(float value, const std::pair<float, float> & range) {
    return std::max(range.first, std::min(range.second, value));
}

} // namespace

bool server_decode_controller_load_artifact_payload(
        const json & payload,
        server_decode_controller_artifact * out_artifact,
        std::string * out_error) {
    if (out_artifact == nullptr) {
        if (out_error) {
            *out_error = "decode controller artifact output must not be null";
        }
        return false;
    }
    try {
        if (json_value(payload, "schema_version", std::string()) != "vicuna.decode_controller_artifact.v1") {
            if (out_error) {
                *out_error = "unsupported decode controller artifact schema";
            }
            return false;
        }

        server_decode_controller_artifact artifact = {};
        artifact.schema_version = json_value(payload, "schema_version", std::string());
        artifact.controller_version = json_value(payload, "controller_version", std::string());
        artifact.input_dimension = json_value(payload.value("input_schema", json::object()), "input_dimension", int32_t(0));
        artifact.hidden_dim = json_value(payload.value("architecture", json::object()), "hidden_dim", int32_t(0));
        if (!parse_float_vector(payload["normalization"]["input_mean"], &artifact.input_mean) ||
                !parse_float_vector(payload["normalization"]["input_std"], &artifact.input_std)) {
            if (out_error) {
                *out_error = "invalid decode controller normalization vectors";
            }
            return false;
        }
        if (artifact.input_dimension <= 0 || artifact.hidden_dim <= 0 ||
                artifact.input_mean.size() != static_cast<size_t>(artifact.input_dimension) ||
                artifact.input_std.size() != static_cast<size_t>(artifact.input_dimension)) {
            if (out_error) {
                *out_error = "invalid decode controller dimensions";
            }
            return false;
        }

        const json action_schema = payload.value("action_schema", json::object());
        artifact.bool_fields = action_schema.value("boolean_fields", std::vector<std::string>());
        artifact.numeric_fields = action_schema.value("numeric_fields", std::vector<std::string>());
        const json numeric_ranges = action_schema.value("numeric_ranges", json::object());
        for (auto it = numeric_ranges.begin(); it != numeric_ranges.end(); ++it) {
            std::vector<float> bounds;
            if (!parse_float_vector(it.value(), &bounds) || bounds.size() != 2) {
                if (out_error) {
                    *out_error = "invalid decode controller numeric range";
                }
                return false;
            }
            artifact.numeric_ranges[it.key()] = {bounds[0], bounds[1]};
        }
        const json profile_vocabs = action_schema.value("profile_vocabs", json::object());
        for (auto it = profile_vocabs.begin(); it != profile_vocabs.end(); ++it) {
            artifact.profile_vocabs[it.key()] = it.value().get<std::vector<std::string>>();
        }

        const json state_dict = payload.value("state_dict", json::object());
        if (!parse_float_matrix(state_dict.at("gru.weight_ih_l0"), artifact.hidden_dim * 3, artifact.input_dimension, &artifact.gru_weight_ih) ||
                !parse_float_matrix(state_dict.at("gru.weight_hh_l0"), artifact.hidden_dim * 3, artifact.hidden_dim, &artifact.gru_weight_hh) ||
                !parse_float_vector(state_dict.at("gru.bias_ih_l0"), &artifact.gru_bias_ih) ||
                !parse_float_vector(state_dict.at("gru.bias_hh_l0"), &artifact.gru_bias_hh)) {
            if (out_error) {
                *out_error = "invalid GRU state tensors";
            }
            return false;
        }
        if (artifact.gru_bias_ih.size() != static_cast<size_t>(artifact.hidden_dim * 3) ||
                artifact.gru_bias_hh.size() != static_cast<size_t>(artifact.hidden_dim * 3)) {
            if (out_error) {
                *out_error = "decode controller GRU bias dimensions mismatch";
            }
            return false;
        }

        const auto parse_head = [](const json & state_dict, const std::string & prefix, size_t out_dim, size_t in_dim, server_decode_controller_linear_head * out_head) -> bool {
            return parse_float_matrix(state_dict.at(prefix + ".weight"), out_dim, in_dim, &out_head->weights) &&
                    parse_float_vector(state_dict.at(prefix + ".bias"), &out_head->bias) &&
                    out_head->bias.size() == out_dim;
        };
        if (!parse_head(state_dict, "bool_head", artifact.bool_fields.size(), artifact.hidden_dim, &artifact.bool_head) ||
                !parse_head(state_dict, "numeric_head", artifact.numeric_fields.size(), artifact.hidden_dim, &artifact.numeric_head)) {
            if (out_error) {
                *out_error = "invalid decode controller output heads";
            }
            return false;
        }
        for (const auto & entry : artifact.profile_vocabs) {
            server_decode_controller_linear_head head = {};
            if (!parse_head(state_dict, "profile_heads." + entry.first, entry.second.size(), artifact.hidden_dim, &head)) {
                if (out_error) {
                    *out_error = "invalid decode controller profile head";
                }
                return false;
            }
            artifact.profile_heads[entry.first] = std::move(head);
        }

        *out_artifact = std::move(artifact);
        return true;
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = e.what();
        }
        return false;
    }
}

bool server_decode_controller_load_artifact(
        const std::string & path,
        server_decode_controller_artifact * out_artifact,
        std::string * out_error) {
    try {
        std::ifstream input(path);
        if (!input.is_open()) {
            if (out_error) {
                *out_error = "failed to open decode controller artifact";
            }
            return false;
        }
        const json payload = json::parse(input, nullptr, true, true);
        return server_decode_controller_load_artifact_payload(payload, out_artifact, out_error);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = e.what();
        }
        return false;
    }
}

bool server_decode_controller_predict(
        const server_decode_controller_artifact & artifact,
        const server_decode_control_observation & observation,
        std::vector<float> * io_hidden_state,
        llama_runtime_control_action * out_action,
        server_decode_control_candidate_metadata * out_metadata,
        std::string * out_error) {
    if (out_action == nullptr || out_metadata == nullptr || io_hidden_state == nullptr) {
        if (out_error) {
            *out_error = "decode controller prediction outputs must not be null";
        }
        return false;
    }
    std::vector<float> input = flatten_observation_features(observation, true);
    if (input.size() != static_cast<size_t>(artifact.input_dimension)) {
        input = flatten_observation_features(observation, false);
    }
    if (input.size() != static_cast<size_t>(artifact.input_dimension)) {
        if (out_error) {
            *out_error = "decode controller input dimension mismatch";
        }
        return false;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = (input[i] - artifact.input_mean[i]) / clamp_nonzero_std(artifact.input_std[i]);
    }
    if (io_hidden_state->size() != static_cast<size_t>(artifact.hidden_dim)) {
        io_hidden_state->assign(artifact.hidden_dim, 0.0f);
    }
    std::vector<float> h_prev = *io_hidden_state;
    std::vector<float> gates(artifact.hidden_dim * 3, 0.0f);
    for (size_t row = 0; row < gates.size(); ++row) {
        float total = artifact.gru_bias_ih[row] + artifact.gru_bias_hh[row];
        for (size_t col = 0; col < input.size(); ++col) {
            total += artifact.gru_weight_ih[row][col] * input[col];
        }
        for (size_t col = 0; col < h_prev.size(); ++col) {
            total += artifact.gru_weight_hh[row][col] * h_prev[col];
        }
        gates[row] = total;
    }
    std::vector<float> h_next(artifact.hidden_dim, 0.0f);
    for (int32_t i = 0; i < artifact.hidden_dim; ++i) {
        const float r = sigmoid_local(gates[i]);
        const float z = sigmoid_local(gates[artifact.hidden_dim + i]);
        float n_affine = artifact.gru_bias_ih[(artifact.hidden_dim * 2) + i] +
                artifact.gru_bias_hh[(artifact.hidden_dim * 2) + i];
        for (size_t col = 0; col < input.size(); ++col) {
            n_affine += artifact.gru_weight_ih[(artifact.hidden_dim * 2) + i][col] * input[col];
        }
        float recurrent = 0.0f;
        for (size_t col = 0; col < h_prev.size(); ++col) {
            recurrent += artifact.gru_weight_hh[(artifact.hidden_dim * 2) + i][col] * h_prev[col];
        }
        const float n = std::tanh(n_affine + r * recurrent);
        h_next[i] = (1.0f - z) * n + z * h_prev[i];
    }
    *io_hidden_state = h_next;

    llama_runtime_control_action action = {};
    action.schema_version = 1;
    action.stage = LLAMA_RUNTIME_CONTROL_STAGE_DECODE_BOUNDARY;
    std::snprintf(action.proposal_source, sizeof(action.proposal_source), "%s", "decode_gru_artifact");
    std::snprintf(action.summary, sizeof(action.summary), "%s", "decode_gru_artifact");

    const auto bool_logits = linear_forward(artifact.bool_head, h_next);
    double confidence_sum = 0.0;
    int confidence_count = 0;
    for (size_t i = 0; i < artifact.bool_fields.size() && i < bool_logits.size(); ++i) {
        const float prob = sigmoid_local(bool_logits[i]);
        const bool value = prob >= 0.5f;
        apply_action_boolean(&action, artifact.bool_fields[i], value);
        confidence_sum += value ? prob : (1.0f - prob);
        ++confidence_count;
    }
    const auto numeric_values = linear_forward(artifact.numeric_head, h_next);
    for (size_t i = 0; i < artifact.numeric_fields.size() && i < numeric_values.size(); ++i) {
        const auto range_it = artifact.numeric_ranges.find(artifact.numeric_fields[i]);
        const float value = range_it != artifact.numeric_ranges.end() ?
                clamp_numeric(numeric_values[i], range_it->second) :
                numeric_values[i];
        apply_action_numeric(&action, artifact.numeric_fields[i], value);
    }
    for (const auto & entry : artifact.profile_heads) {
        float profile_confidence = 0.0f;
        const auto logits = linear_forward(entry.second, h_next);
        const auto vocab_it = artifact.profile_vocabs.find(entry.first);
        if (vocab_it == artifact.profile_vocabs.end()) {
            continue;
        }
        const std::string selected = argmax_vocab(logits, vocab_it->second, &profile_confidence);
        confidence_sum += profile_confidence;
        ++confidence_count;
        if (entry.first == "structure.grammar_profile_id") {
            std::snprintf(action.structure.grammar_profile_id, sizeof(action.structure.grammar_profile_id), "%s", selected.c_str());
        } else if (entry.first == "structure.logit_bias_profile_id") {
            std::snprintf(action.structure.logit_bias_profile_id, sizeof(action.structure.logit_bias_profile_id), "%s", selected.c_str());
        } else if (entry.first == "steering.cvec_profile_id") {
            std::snprintf(action.steering.cvec_profile_id, sizeof(action.steering.cvec_profile_id), "%s", selected.c_str());
        }
    }

    if (!observation.mask.allow_sampling) {
        action.sampling = {};
    }
    if (!observation.mask.allow_repetition) {
        action.repetition = {};
    }
    if (!observation.mask.allow_structure) {
        action.structure = {};
    }
    if (!observation.mask.allow_branch) {
        action.branch = {};
    }
    if (!observation.mask.allow_steering) {
        action.steering = {};
    }
    action.branch.branch_sample_count = std::min(action.branch.branch_sample_count, observation.mask.max_branch_sample_count);
    action.valid = true;

    *out_action = action;
    out_metadata->available = true;
    out_metadata->controller_version = artifact.controller_version;
    out_metadata->controller_alias.clear();
    out_metadata->confidence_present = true;
    out_metadata->confidence = confidence_count > 0 ? static_cast<float>(confidence_sum / static_cast<double>(confidence_count)) : 0.0f;
    out_metadata->proposal_source = "decode_gru_artifact";
    return true;
}

bool server_decode_controller_infer_callback(
        const server_decode_control_observation * observation,
        llama_runtime_control_action * out_action,
        server_decode_control_candidate_metadata * out_metadata,
        void * user_data) {
    if (observation == nullptr || out_action == nullptr || out_metadata == nullptr || user_data == nullptr) {
        return false;
    }
    auto * state = static_cast<server_llama_runtime_control_state *>(user_data);
    if (!state->decode_controller) {
        return false;
    }
    std::string error;
    return server_decode_controller_predict(
            *state->decode_controller,
            *observation,
            &state->decode_controller_hidden,
            out_action,
            out_metadata,
            &error);
}
