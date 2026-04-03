#include "server-cvec-generator.h"

#include <cmath>
#include <fstream>
#include <numeric>

namespace {

static float clamp_nonzero_std(float value) {
    return std::fabs(value) < 1e-6f ? 1.0f : value;
}

static std::vector<float> flatten_cvec_input(
        const server_emotive_vector & moment,
        const server_emotive_vad & vad) {
    return {
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
    };
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

static float vector_norm(const std::vector<float> & values) {
    double total = 0.0;
    for (float value : values) {
        total += static_cast<double>(value) * static_cast<double>(value);
    }
    return static_cast<float>(std::sqrt(total));
}

} // namespace

bool server_cvec_generator_load_artifact_payload(
        const json & payload,
        server_cvec_generator_artifact * out_artifact,
        std::string * out_error) {
    if (out_artifact == nullptr) {
        if (out_error) {
            *out_error = "cvec artifact output must not be null";
        }
        return false;
    }

    try {
        if (json_value(payload, "schema_version", std::string()) != "vicuna.cvec_generator_artifact.v1") {
            if (out_error) {
                *out_error = "unsupported cvec generator artifact schema";
            }
            return false;
        }

        server_cvec_generator_artifact artifact = {};
        artifact.schema_version = json_value(payload, "schema_version", std::string());
        artifact.generator_version = json_value(payload, "generator_version", std::string());
        artifact.target_embedding_dim = json_value(payload, "target_embedding_dim", int32_t(0));
        artifact.target_layer_start = json_value(payload, "target_layer_start", int32_t(0));
        artifact.target_layer_end = json_value(payload, "target_layer_end", int32_t(-1));
        const json architecture = payload.value("architecture", json::object());
        artifact.activation = json_value(architecture, "activation", std::string("tanh"));
        const json normalization = payload.value("normalization", json::object());
        artifact.output_mode = json_value(normalization, "output_mode", std::string("none"));
        artifact.output_norm_cap = json_value(normalization, "output_norm_cap", 0.0f);

        if (!parse_float_vector(normalization.at("input_mean"), &artifact.input_mean) ||
                !parse_float_vector(normalization.at("input_std"), &artifact.input_std)) {
            if (out_error) {
                *out_error = "invalid cvec normalization vectors";
            }
            return false;
        }

        if (artifact.input_mean.size() != 17 || artifact.input_std.size() != 17) {
            if (out_error) {
                *out_error = "cvec generator expects 17 EM/VAD input dimensions";
            }
            return false;
        }

        const json layers = payload.value("layers", json::array());
        if (!layers.is_array() || layers.empty()) {
            if (out_error) {
                *out_error = "cvec generator artifact missing layers";
            }
            return false;
        }
        artifact.layers.clear();
        artifact.layers.reserve(layers.size());
        size_t expected_input_dim = artifact.input_mean.size();
        for (const auto & layer_json : layers) {
            if (!layer_json.is_object() || !layer_json.contains("weights") || !layer_json.contains("bias")) {
                if (out_error) {
                    *out_error = "invalid cvec generator layer";
                }
                return false;
            }
            server_cvec_generator_layer layer = {};
            if (!parse_float_vector(layer_json.at("bias"), &layer.bias)) {
                if (out_error) {
                    *out_error = "invalid cvec generator bias vector";
                }
                return false;
            }
            const json weights = layer_json.at("weights");
            if (!weights.is_array() || weights.size() != layer.bias.size()) {
                if (out_error) {
                    *out_error = "cvec generator weight rows do not match bias size";
                }
                return false;
            }
            layer.weights.reserve(weights.size());
            for (const auto & row_json : weights) {
                std::vector<float> row;
                if (!parse_float_vector(row_json, &row) || row.size() != expected_input_dim) {
                    if (out_error) {
                        *out_error = "cvec generator weight row has invalid width";
                    }
                    return false;
                }
                layer.weights.push_back(std::move(row));
            }
            expected_input_dim = layer.bias.size();
            artifact.layers.push_back(std::move(layer));
        }

        if (expected_input_dim != static_cast<size_t>(artifact.target_embedding_dim)) {
            if (out_error) {
                *out_error = "cvec generator output width does not match target_embedding_dim";
            }
            return false;
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

bool server_cvec_generator_load_artifact(
        const std::string & path,
        server_cvec_generator_artifact * out_artifact,
        std::string * out_error) {
    try {
        std::ifstream input(path);
        if (!input.is_open()) {
            if (out_error) {
                *out_error = "failed to open cvec artifact";
            }
            return false;
        }
        const json payload = json::parse(input, nullptr, true, true);
        return server_cvec_generator_load_artifact_payload(payload, out_artifact, out_error);
    } catch (const std::exception & e) {
        if (out_error) {
            *out_error = e.what();
        }
        return false;
    }
}

bool server_cvec_generator_infer(
        const server_cvec_generator_artifact & artifact,
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        server_cvec_generator_result * out_result,
        std::string * out_error) {
    if (out_result == nullptr) {
        if (out_error) {
            *out_error = "cvec inference result must not be null";
        }
        return false;
    }

    std::vector<float> hidden = flatten_cvec_input(moment, vad);
    if (hidden.size() != artifact.input_mean.size() || hidden.size() != artifact.input_std.size()) {
        if (out_error) {
            *out_error = "cvec generator input normalization shape mismatch";
        }
        return false;
    }
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = (hidden[i] - artifact.input_mean[i]) / clamp_nonzero_std(artifact.input_std[i]);
    }

    for (size_t layer_index = 0; layer_index < artifact.layers.size(); ++layer_index) {
        const auto & layer = artifact.layers[layer_index];
        std::vector<float> next(layer.bias.size(), 0.0f);
        for (size_t row = 0; row < layer.weights.size(); ++row) {
            float total = layer.bias[row];
            for (size_t col = 0; col < layer.weights[row].size(); ++col) {
                total += hidden[col] * layer.weights[row][col];
            }
            next[row] = total;
        }
        const bool is_last = layer_index + 1 == artifact.layers.size();
        if (!is_last) {
            if (artifact.activation == "relu") {
                for (float & value : next) {
                    value = std::max(0.0f, value);
                }
            } else {
                for (float & value : next) {
                    value = std::tanh(value);
                }
            }
        }
        hidden = std::move(next);
    }

    if (artifact.output_mode == "tanh") {
        for (float & value : hidden) {
            value = std::tanh(value);
        }
    }

    server_cvec_generator_result result = {};
    result.vector = std::move(hidden);
    result.norm = vector_norm(result.vector);
    if (artifact.output_norm_cap > 0.0f && result.norm > artifact.output_norm_cap) {
        const float scale = artifact.output_norm_cap / std::max(result.norm, 1e-6f);
        for (float & value : result.vector) {
            value *= scale;
        }
        result.norm = vector_norm(result.vector);
        result.clipped = true;
    }

    *out_result = std::move(result);
    return true;
}
