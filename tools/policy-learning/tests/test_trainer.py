#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from policy_dataset import append_transitions
from policy_trainer import train_policy
from policy_training_contract import build_training_corpus


def make_transition(transition_id: str, available_tool_count: int) -> dict:
    selected_mode = "direct" if available_tool_count == 0 else "tool_light"
    return {
        "transition_id": transition_id,
        "request_id": f"req-{transition_id}",
        "decision_id": f"dec-{transition_id}",
        "created_at_ms": 1700000000000,
        "observation": {
            "request_id": f"req-{transition_id}",
            "mode_label": "chat",
            "bridge_scoped": False,
            "cognitive_replay": False,
            "heuristic_matched": False,
            "available_tool_count": available_tool_count,
            "parallel_tool_calls_requested": available_tool_count > 0,
            "input_message_count": 1,
            "vad": {"valence": 0.1, "arousal": 0.2, "dominance": 0.3},
        },
        "action_mask": {
            "allowed_modes": ["direct", "tool_light"],
            "allowed_reasoning_depths": ["short", "medium"],
            "allowed_response_budget_buckets": [256, 512],
            "allowed_reasoning_budget_buckets": [0, 64],
            "max_tool_parallelism_cap": 1,
            "allow_interrupt": True,
            "allow_replan": True,
            "allow_early_stop": True,
            "allow_force_synthesis": True,
        },
        "executed_action": {
            "selected_mode": selected_mode,
            "reasoning_depth": "short",
            "response_budget_bucket": 512,
            "reasoning_budget_bucket": 64,
            "token_budget_bucket": 512,
            "tool_parallelism_cap": 0,
            "interrupt_allowed": False,
            "replan_required": False,
            "early_stop_ok": True,
            "force_synthesis": False,
        },
        "reward_model": {
            "schema_version": "policy_reward_model_v1",
            "model_version": "desired_state_reward_v1",
        },
        "reward_breakdown": {
            "schema_version": "policy_reward_breakdown_v1",
            "model_version": "desired_state_reward_v1",
            "before_score": 0.45,
            "after_score": 0.72,
            "progress_reward": 0.18,
            "terminal_closeness_reward": 0.14,
            "completion_quality_reward": 0.20,
            "latency_cost": -0.05,
            "token_cost": -0.05,
            "tool_success_reward": 0.0,
            "candidate_failure_penalty": 0.0,
            "total": 1.0,
        },
        "reward_events": [{"kind": "completion_quality", "value": 1.0, "weight": 1.0, "source": "test"}],
        "reward_total": 1.0,
        "terminated": True,
        "termination_reason": "stop",
    }


def bootstrap_dataset(dataset_dir: Path) -> None:
    append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload={
            "behavior_policy_version": "control_surface_v2",
            "candidate_policy_version": None,
            "reward_model": {
                "schema_version": "policy_reward_model_v1",
                "model_version": "desired_state_reward_v1",
            },
        },
        transitions=[
            make_transition("tr-1", 0),
            make_transition("tr-2", 1),
        ],
    )
    build_training_corpus(dataset_dir)


def test_train_policy_is_deterministic(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    bootstrap_dataset(dataset_dir)

    first = train_policy(
        dataset_dir=dataset_dir,
        model_name="vicuna-governance",
        run_root=tmp_path / "runs-first",
    )
    second = train_policy(
        dataset_dir=dataset_dir,
        model_name="vicuna-governance",
        run_root=tmp_path / "runs-second",
    )

    first_artifact = Path(first["artifact_path"]).read_text(encoding="utf-8")
    second_artifact = Path(second["artifact_path"]).read_text(encoding="utf-8")
    assert first["policy_version"] == second["policy_version"]
    assert first_artifact == second_artifact
    assert first["metrics"]["exact_match_rate"] == 1.0
    assert second["metrics"]["exact_match_rate"] == 1.0


def test_train_policy_writes_manifest_and_artifact(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    bootstrap_dataset(dataset_dir)

    result = train_policy(
        dataset_dir=dataset_dir,
        model_name="vicuna-governance",
        run_root=tmp_path / "runs",
    )

    manifest = json.loads(
        Path(result["training_run_manifest_path"]).read_text(encoding="utf-8")
    )
    artifact = json.loads(Path(result["artifact_path"]).read_text(encoding="utf-8"))
    assert manifest["policy_version"] == artifact["policy_version"]
    assert manifest["record_count"] == 2
    assert manifest["reward_model_version"] == "desired_state_reward_v1"
    assert artifact["training_metrics"]["record_count"] == 2
    assert artifact["reward_model_version"] == "desired_state_reward_v1"
    assert "thinking_mode" in artifact["global_action_priors"]
    assert "prefix_profile" in artifact["global_action_priors"]
