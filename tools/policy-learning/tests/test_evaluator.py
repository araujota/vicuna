#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from policy_dataset import append_transitions
from policy_evaluator import evaluate_dataset


def make_transition(transition_id: str, request_id: str, available_tool_count: int) -> dict:
    return {
        "transition_id": transition_id,
        "request_id": request_id,
        "decision_id": f"dec-{request_id}",
        "created_at_ms": 1700000000000,
        "observation": {
            "request_id": request_id,
            "available_tool_count": available_tool_count,
            "heuristic_matched": False,
        },
        "action_mask": {
            "allowed_modes": ["direct", "tool_light"],
            "allowed_reasoning_depths": ["short", "medium"],
            "max_tool_parallelism_cap": 1,
            "allow_interrupt": True,
            "allow_replan": True,
            "allow_early_stop": True,
            "allow_force_synthesis": True,
        },
        "executed_action": {
            "selected_mode": "direct" if available_tool_count == 0 else "tool_light",
            "reasoning_depth": "short",
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
        "reward_events": [{"kind": "completion_quality", "value": 1.0, "weight": 1.0, "source": "test"}],
        "reward_total": 1.0,
        "terminated": True,
        "termination_reason": "stop",
    }


def test_evaluate_dataset_with_command_adapter(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    status_payload = {
        "behavior_policy_version": "control_surface_v2",
        "candidate_policy_version": None,
        "reward_model": {
            "schema_version": "policy_reward_model_v1",
            "model_version": "desired_state_reward_v1",
        },
    }
    append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload=status_payload,
        transitions=[
            make_transition("tr-1", "req-1", 0),
            make_transition("tr-2", "req-2", 1),
        ],
    )

    candidate_script = TOOLS_ROOT / "example_candidate_policy.py"
    report = evaluate_dataset(
        dataset_dir=dataset_dir,
        candidate_command=f"{sys.executable} {candidate_script}",
    )

    assert report["evaluated_records"] == 2
    assert report["successful_predictions"] == 2
    assert report["candidate_failure_count"] == 0
    assert report["invalid_action_count"] == 0
    assert "exact_match_rate" in report
    assert "thinking_mode_match_rate" in report
    assert "prefix_profile_match_rate" in report
    assert "per_head_disagreement" in report
    assert report["reward_model_version"] == "desired_state_reward_v1"
