#pragma once

#include "server-common.h"
#include "server-emotive-runtime.h"

#include <string>
#include <vector>

struct server_cvec_generator_layer {
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
};

struct server_cvec_generator_artifact {
    std::string schema_version;
    std::string generator_version;
    std::string activation = "tanh";
    std::string output_mode = "none";
    int32_t target_embedding_dim = 0;
    int32_t target_layer_start = 0;
    int32_t target_layer_end = -1;
    float output_norm_cap = 0.0f;
    std::vector<float> input_mean;
    std::vector<float> input_std;
    std::vector<server_cvec_generator_layer> layers;
};

struct server_cvec_generator_result {
    std::vector<float> vector;
    float norm = 0.0f;
    bool clipped = false;
};

bool server_cvec_generator_load_artifact(
        const std::string & path,
        server_cvec_generator_artifact * out_artifact,
        std::string * out_error);

bool server_cvec_generator_load_artifact_payload(
        const json & payload,
        server_cvec_generator_artifact * out_artifact,
        std::string * out_error);

bool server_cvec_generator_infer(
        const server_cvec_generator_artifact & artifact,
        const server_emotive_vector & moment,
        const server_emotive_vad & vad,
        server_cvec_generator_result * out_result,
        std::string * out_error);
