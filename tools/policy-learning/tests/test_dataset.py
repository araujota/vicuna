#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from policy_dataset import (
    append_decode_traces,
    append_transitions,
    load_dataset_rows,
    load_decode_trace_rows,
    load_manifest,
)


def make_transition(transition_id: str, request_id: str) -> dict:
    return {
        "transition_id": transition_id,
        "request_id": request_id,
        "decision_id": f"dec-{request_id}",
        "created_at_ms": 1700000000000,
        "observation": {"request_id": request_id},
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
            "selected_mode": "direct",
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
        "reward_breakdown": {
            "schema_version": "policy_reward_breakdown_v1",
            "model_version": "desired_state_reward_v1",
            "before_score": 0.40,
            "after_score": 0.70,
            "progress_reward": 0.20,
            "terminal_closeness_reward": 0.10,
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


def test_append_transitions_creates_manifest_and_dedupes(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    status_payload = {
        "behavior_policy_version": "control_surface_v2",
        "candidate_policy_version": None,
        "reward_model": {
            "schema_version": "policy_reward_model_v1",
            "model_version": "desired_state_reward_v1",
        },
    }
    transitions = [
        make_transition("tr-1", "req-1"),
        make_transition("tr-2", "req-2"),
    ]

    first = append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload=status_payload,
        transitions=transitions,
    )
    second = append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload=status_payload,
        transitions=transitions,
    )

    assert first["appended_count"] == 2
    assert second["appended_count"] == 0
    manifest = load_manifest(dataset_dir)
    assert manifest is not None
    assert manifest["stored_transition_count"] == 2
    assert manifest["source_reward_model_version"] == "desired_state_reward_v1"
    rows = load_dataset_rows(dataset_dir)
    assert len(rows) == 2
    assert rows[0]["transition"]["request_id"] == "req-1"


def test_append_decode_traces_creates_separate_store_and_dedupes(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    status_payload = {
        "behavior_policy_version": "control_surface_v2",
        "candidate_policy_version": None,
        "reward_model": {
            "schema_version": "policy_reward_model_v1",
            "model_version": "desired_state_reward_v1",
        },
    }
    traces = [
        {
            "request_id": "req-1",
            "emotive_trace_id": "trace-1",
            "created_at_ms": 1700000001000,
            "step_count": 1,
            "steps": [
                {
                    "step_index": 1,
                    "input_features": [0.1, 0.2],
                }
            ],
        }
    ]
    first = append_decode_traces(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload=status_payload,
        decode_traces=traces,
    )
    second = append_decode_traces(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload=status_payload,
        decode_traces=traces,
    )
    assert first["appended_count"] == 1
    assert second["appended_count"] == 0
    manifest = load_manifest(dataset_dir)
    assert manifest is not None
    assert manifest["stored_decode_trace_count"] == 1
    rows = load_decode_trace_rows(dataset_dir)
    assert len(rows) == 1
    assert rows[0]["decode_trace"]["request_id"] == "req-1"
