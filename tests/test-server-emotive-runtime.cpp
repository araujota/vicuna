#include "tools/server/server-emotive-runtime.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>

static server_emotive_block_record eval_runtime(
        server_emotive_runtime & runtime,
        const std::string & payload,
        const server_emotive_vector * previous_moment = nullptr,
        const server_emotive_vad * previous_vad = nullptr) {
    const std::vector<float> none;
    return runtime.evaluate_block(
            SERVER_EMOTIVE_BLOCK_RUNTIME_EVENT,
            payload,
            0,
            none,
            none,
            previous_moment,
            previous_vad);
}

int main() {
    setenv("VICUNA_NATIVE_RESPONSE_BUDGET_BUCKET", "512", 1);
    setenv("VICUNA_NATIVE_REASONING_BUDGET_BUCKET", "512", 1);

    server_emotive_runtime_config config;
    config.enabled = true;
    config.embedding.enabled = false;
    server_emotive_runtime runtime(config);

    server_metacognitive_control_state control_state = {};
    control_state.bridge_scoped = true;
    const auto starter_policy = runtime.compute_control_policy(control_state);
    assert(starter_policy.response_budget_bucket == 512);
    assert(starter_policy.reasoning_budget_bucket == 512);
    const server_policy_action starter_action = {};
    assert(starter_action.response_budget_bucket == 512);
    assert(starter_action.reasoning_budget_bucket == 512);

    server_request_policy_config request_policy = {};
    request_policy.response_budget_bucket = 512;
    request_policy.reasoning_budget_bucket = 512;
    request_policy.token_budget_bucket = 512;
    request_policy.thinking_enabled = true;
    request_policy.thinking_mode = "enabled";
    request_policy.temperature_present = true;
    request_policy.temperature = 0.2;
    request_policy.top_k_present = true;
    request_policy.top_k = 24;
    request_policy.top_p_present = true;
    request_policy.top_p = 0.95;
    request_policy.min_p_present = true;
    request_policy.min_p = 0.05;
    const json request_policy_json = server_request_policy_config_to_json(request_policy);
    server_request_policy_config parsed_request_policy = {};
    assert(server_request_policy_config_from_json(request_policy_json, &parsed_request_policy, nullptr));
    assert(parsed_request_policy.response_budget_bucket == 512);
    assert(parsed_request_policy.top_k == 24);
    assert(std::fabs(parsed_request_policy.min_p - 0.05) < 1e-6);

    const std::string high_uncertainty = R"json(
{
  "type": "decode_step_completed",
  "timestamp_us": 1000000,
  "candidate_count": 256,
  "mean_entropy": 5.7,
  "max_entropy": 5.9,
  "mean_margin": 0.05,
  "distribution": {"sampled_prob": 0.16, "stop_prob": 0.02, "candidate_count": 256},
  "agreement_score": 0.10,
  "consistency_entropy": 0.82,
  "branch_disagreement": 0.74,
  "verifier_disagreement": 0.58,
  "verifier_active": true,
  "attention_availability": "full",
  "attention_entropy_mean": 4.2,
  "attention_entropy_max": 4.5,
  "attention_top1_mass_mean": 0.08,
  "attention_top1_mass_max": 0.12,
  "expert_count": 16,
  "route_entropy_mean": 2.3,
  "route_entropy_max": 2.6,
  "route_top1_weight_mean": 0.24,
  "route_top1_weight_max": 0.28,
  "sampler_profile": {"temperature": 1.1, "top_p": 0.98, "min_p": 0.0, "typical_p": 1.0},
  "repeat_hit_rate": 0.18
}
)json";

    const auto uncertain = eval_runtime(runtime, high_uncertainty);
    assert(uncertain.runtime_signals.available);
    assert(uncertain.runtime_signals.timestamp_ms == 1000);
    assert(uncertain.runtime_signals.mean_entropy > 0.9f);
    assert(uncertain.runtime_signals.mean_margin < 0.1f);
    assert(uncertain.moment.epistemic_pressure > 0.55f);
    assert(uncertain.moment.caution > 0.45f);
    assert(uncertain.moment.confidence < 0.35f);
    assert(uncertain.vad.dominance < 0.0f);

    const std::string stable_success = R"json(
{
  "type": "token_accepted",
  "timestamp_us": 1200000,
  "candidate_count": 256,
  "mean_entropy": 0.9,
  "max_entropy": 0.9,
  "mean_margin": 3.5,
  "distribution": {"sampled_prob": 0.94, "stop_prob": 0.01, "candidate_count": 256},
  "agreement_score": 0.94,
  "consistency_entropy": 0.08,
  "branch_disagreement": 0.04,
  "verifier_disagreement": 0.02,
  "attention_availability": "full",
  "attention_entropy_mean": 0.7,
  "attention_entropy_max": 0.9,
  "attention_top1_mass_mean": 0.83,
  "attention_top1_mass_max": 0.88,
  "expert_count": 16,
  "route_entropy_mean": 0.5,
  "route_entropy_max": 0.7,
  "route_top1_weight_mean": 0.89,
  "route_top1_weight_max": 0.92,
  "dominant_expert_fractions": [0.74, 0.18],
  "repeat_hit_rate": 0.02,
  "tool_correctness_score": 1.0,
  "tool_correctness_confidence": 1.0,
  "prompt_section_label": "answer"
}
)json";

    const auto stable = eval_runtime(runtime, stable_success, &uncertain.moment, &uncertain.vad);
    assert(stable.moment.confidence > uncertain.moment.confidence);
    assert(stable.moment.runtime_trust > uncertain.moment.runtime_trust);
    assert(stable.moment.satisfaction > uncertain.moment.satisfaction);
    assert(stable.moment.planning_clarity > uncertain.moment.planning_clarity);
    assert(stable.vad.dominance > uncertain.vad.dominance);

    const std::string runtime_fault = R"json(
{
  "type": "runtime_fault",
  "timestamp_us": 1500000,
  "runtime_failure": true,
  "fault_type": "sampler_failed",
  "status_code": 1,
  "tool_correctness_score": 0.0,
  "tool_correctness_confidence": 1.0,
  "verifier_active": true,
  "verifier_disagreement": 0.40,
  "branch_disagreement": 0.35
}
)json";

    const auto failed = eval_runtime(runtime, runtime_fault, &stable.moment, &stable.vad);
    assert(failed.moment.runtime_failure_pressure > stable.moment.runtime_failure_pressure);
    assert(failed.moment.runtime_trust < stable.moment.runtime_trust);
    assert(failed.moment.frustration > stable.moment.frustration);
    assert(failed.vad.dominance < stable.vad.dominance);

    unsetenv("VICUNA_NATIVE_RESPONSE_BUDGET_BUCKET");
    unsetenv("VICUNA_NATIVE_REASONING_BUDGET_BUCKET");

    return 0;
}
