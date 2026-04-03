#pragma once

#include "server-runtime-control.h"

#include <map>

struct server_decode_controller_linear_head {
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
};

struct server_decode_controller_artifact {
    std::string schema_version;
    std::string controller_version;
    int32_t input_dimension = 0;
    int32_t hidden_dim = 0;
    std::vector<float> input_mean;
    std::vector<float> input_std;
    std::vector<std::vector<float>> gru_weight_ih;
    std::vector<std::vector<float>> gru_weight_hh;
    std::vector<float> gru_bias_ih;
    std::vector<float> gru_bias_hh;
    server_decode_controller_linear_head bool_head;
    server_decode_controller_linear_head numeric_head;
    std::map<std::string, server_decode_controller_linear_head> profile_heads;
    std::vector<std::string> bool_fields;
    std::vector<std::string> numeric_fields;
    std::map<std::string, std::pair<float, float>> numeric_ranges;
    std::map<std::string, std::vector<std::string>> profile_vocabs;
};

bool server_decode_controller_load_artifact(
        const std::string & path,
        server_decode_controller_artifact * out_artifact,
        std::string * out_error);

bool server_decode_controller_load_artifact_payload(
        const json & payload,
        server_decode_controller_artifact * out_artifact,
        std::string * out_error);

bool server_decode_controller_predict(
        const server_decode_controller_artifact & artifact,
        const server_decode_control_observation & observation,
        std::vector<float> * io_hidden_state,
        llama_runtime_control_action * out_action,
        server_decode_control_candidate_metadata * out_metadata,
        std::string * out_error);

bool server_decode_controller_infer_callback(
        const server_decode_control_observation * observation,
        llama_runtime_control_action * out_action,
        server_decode_control_candidate_metadata * out_metadata,
        void * user_data);
