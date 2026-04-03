#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from decode_gru_contract import build_decode_gru_training_corpus, load_decode_gru_training_records
from decode_gru_evaluator import evaluate_decode_gru_controller
from decode_gru_model import infer_decode_controller_action, load_decode_controller_artifact
from decode_gru_trainer import train_decode_gru_controller
from policy_dataset import append_decode_traces


def _trace_step(step_index: int, confidence: float, entropy: float) -> dict:
    return {
        "timestamp_ms": 1700000000000 + step_index,
        "seq_id": 1,
        "step_index": step_index,
        "output_index": step_index - 1,
        "moment": {
            "confidence": confidence,
            "curiosity": 0.2,
            "frustration": 0.1,
            "satisfaction": 0.4,
            "momentum": 0.3,
            "caution": 0.2,
            "stall": 0.1,
            "epistemic_pressure": 0.3,
            "planning_clarity": 0.5,
            "user_alignment": 0.7,
            "semantic_novelty": 0.2,
            "runtime_trust": 0.6,
            "runtime_failure_pressure": 0.1,
            "contradiction_pressure": 0.2,
        },
        "vad": {"valence": 0.1, "arousal": 0.2, "dominance": 0.3},
        "runtime_signals": {
            "mean_entropy": entropy,
            "max_entropy": entropy + 0.1,
            "mean_margin": 0.2,
            "sampled_prob": 0.5,
            "stop_prob": 0.1,
            "repeat_hit_rate": 0.02,
            "route_entropy_mean": 0.1,
            "route_entropy_max": 0.1,
            "route_top1_weight_mean": 0.7,
            "route_top1_weight_max": 0.8,
            "attention_entropy_mean": 0.2,
            "attention_entropy_max": 0.2,
            "attention_top1_mass_mean": 0.6,
            "attention_top1_mass_max": 0.7,
            "agreement_score": 0.8,
            "consistency_entropy": 0.1,
            "branch_disagreement": 0.1,
            "verifier_disagreement": 0.1,
            "graph_value_mean_abs": 0.1,
            "graph_value_rms": 0.1,
            "graph_value_max_abs": 0.1,
            "dominant_expert_fraction_top1": 0.7,
            "dominant_expert_fraction_mass": 0.9,
            "timing_decode_ms": 5.0,
            "timing_sample_ms": 1.0,
            "timing_delta_ms": 6.0,
            "memory_budget_ratio": 0.2,
            "attention_budget_ratio": 0.2,
            "recurrent_budget_ratio": 0.0,
            "candidate_count": 8,
            "attention_pos_min": 0,
            "attention_pos_max": 128,
            "recurrent_pos_min": 0,
            "recurrent_pos_max": 0,
            "expert_count": 8,
            "experts_selected": 2,
            "dominant_expert_count": 2,
            "comparison_count": 1,
            "semantic_group_count": 1,
            "status_code": 0,
            "runtime_failure": False,
            "verifier_active": False,
            "grammar_active": False,
            "logit_bias_active": False,
            "backend_sampler": False,
            "optimized": False,
            "prompt_section_changed": False,
        },
        "bundles": {
            "uncertainty_regulation": 0.3,
            "anti_repetition_recovery": 0.2,
            "structural_validity": 0.4,
            "verification_pressure": 0.2,
            "commit_efficiency": 0.5,
            "steering_pressure": 0.6,
        },
        "decode_policy_config": {
            "schema_version": "decode_policy_config_v1",
            "base_temperature": 0.7,
            "base_top_k": 32,
            "base_top_p": 0.95,
            "base_min_p": 0.0,
            "action_mask": {
                "allow_sampling": True,
                "allow_repetition": True,
                "allow_structure": True,
                "allow_branch": True,
                "allow_steering": True,
                "max_branch_sample_count": 4,
            },
            "control_limits": {
                "min_temperature": 0.1,
                "max_temperature": 1.6,
                "min_top_p": 0.1,
                "max_top_p": 1.0,
                "min_min_p": 0.0,
                "max_min_p": 0.4,
                "min_typical_p": 0.1,
                "max_typical_p": 1.0,
                "max_top_n_sigma": 5.0,
                "min_repeat_penalty": 1.0,
                "max_repeat_penalty": 1.6,
                "max_frequency_penalty": 1.5,
                "max_presence_penalty": 1.5,
                "max_dry_multiplier": 2.0,
                "max_penalty_last_n": 256,
                "max_dry_allowed_length": 32,
                "max_dry_penalty_last_n": 256,
            },
        },
        "mask": {
            "allow_sampling": True,
            "allow_repetition": True,
            "allow_structure": True,
            "allow_branch": True,
            "allow_steering": True,
            "max_branch_sample_count": 4,
        },
        "previous_executed_action_available": step_index > 1,
        "previous_executed_action": {
            "valid": True,
            "sampling": {"enabled": True, "has_temperature": True, "temperature": 0.7, "has_top_k": True, "top_k": 32, "has_top_p": True, "top_p": 0.9},
            "repetition": {"enabled": False},
            "structure": {"enabled": False},
            "branch": {"enabled": False},
            "steering": {"enabled": True, "clear_cvec": False},
        },
        "teacher_action": {
            "valid": True,
            "sampling": {
                "enabled": True,
                "has_temperature": True,
                "temperature": 0.65,
                "has_top_k": True,
                "top_k": 24,
                "has_top_p": True,
                "top_p": 0.92,
                "has_min_p": False,
                "has_typical_p": False,
                "has_top_n_sigma": False,
                "min_keep": 1,
            },
            "repetition": {"enabled": False},
            "structure": {"enabled": False, "clear_grammar": False, "clear_logit_bias": False, "grammar_profile_id": "", "logit_bias_profile_id": ""},
            "branch": {"enabled": False, "checkpoint_now": False, "request_verify": False, "checkpoint_slot": 0, "restore_slot": 0, "branch_sample_count": 0},
            "steering": {"enabled": True, "clear_cvec": False, "cvec_profile_id": "truthfulness"},
        },
        "executed_action": {"valid": True},
        "candidate_metadata": {"available": False},
        "next_outcome": {
            "available": True,
            "d_mean_entropy": -0.1,
            "d_confidence": 0.1,
            "d_stall": -0.05,
        },
    }


def test_decode_gru_contract_train_and_eval(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    append_decode_traces(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload={"behavior_policy_version": "control_surface_v2"},
        decode_traces=[
            {
                "request_id": "req-1",
                "emotive_trace_id": "trace-1",
                "created_at_ms": 1700000000000,
                "step_count": 2,
                "steps": [
                    _trace_step(1, 0.4, 0.7),
                    _trace_step(2, 0.5, 0.5),
                ],
            }
        ],
    )
    built = build_decode_gru_training_corpus(dataset_dir, sequence_length=2, stride=1)
    assert built["record_count"] >= 1
    training_manifest, records = load_decode_gru_training_records(dataset_dir)
    assert training_manifest["input_dimension"] > 10
    assert records[0]["steps"][0]["teacher_target"]["profiles"]["steering.cvec_profile_id"] == "truthfulness"
    assert records[0]["steps"][0]["decode_policy_config"]["base_top_k"] == 32
    assert records[0]["steps"][0]["mask"]["max_branch_sample_count"] == 4

    trained = train_decode_gru_controller(
        dataset_dir=dataset_dir,
        model_name="vicuna-decode",
        hidden_dim=16,
        epochs=2,
        batch_size=1,
    )
    artifact = load_decode_controller_artifact(Path(trained["artifact_path"]))
    prediction = infer_decode_controller_action(
        artifact,
        [records[0]["steps"][0]["input_features"]],
        action_mask=records[0]["steps"][0]["mask"],
    )
    assert "boolean" in prediction
    assert "numeric" in prediction
    assert "profiles" in prediction

    report = evaluate_decode_gru_controller(
        dataset_dir=dataset_dir,
        artifact_path=Path(trained["artifact_path"]),
        report_path=tmp_path / "decode_eval.json",
    )
    assert report["record_count"] >= 1
    assert report["valid_action_rate"] >= 0.0
