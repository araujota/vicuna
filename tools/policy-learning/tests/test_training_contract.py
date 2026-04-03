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
from policy_training_contract import build_training_corpus, build_training_record


def make_transition(response_budget_bucket: int = 1024, reasoning_budget_bucket: int = 1024) -> dict:
    return {
        "transition_id": "tr-1",
        "request_id": "req-1",
        "decision_id": "dec-1",
        "created_at_ms": 1700000000000,
        "observation": {"request_id": "req-1"},
        "action_mask": {
            "allowed_modes": ["direct"],
            "allowed_reasoning_depths": ["short"],
            "allowed_response_budget_buckets": [256, 512, 1024],
            "allowed_reasoning_budget_buckets": [0, 64, 1024],
            "max_tool_parallelism_cap": 0,
            "allow_interrupt": False,
            "allow_replan": False,
            "allow_early_stop": True,
            "allow_force_synthesis": False,
        },
        "executed_action": {
            "selected_mode": "direct",
            "reasoning_depth": "short",
            "response_budget_bucket": response_budget_bucket,
            "token_budget_bucket": response_budget_bucket,
            "reasoning_budget_bucket": reasoning_budget_bucket,
            "tool_parallelism_cap": 0,
            "interrupt_allowed": False,
            "replan_required": False,
            "early_stop_ok": True,
            "force_synthesis": False,
        },
        "reward_events": [{"kind": "completion_quality", "value": 1.0, "weight": 1.0, "source": "test"}],
        "reward_total": 1.0,
        "terminated": True,
        "termination_reason": "stop",
    }


def test_build_training_corpus_writes_records(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload={
            "behavior_policy_version": "control_surface_v2",
            "candidate_policy_version": None,
        },
        transitions=[make_transition()],
    )

    result = build_training_corpus(dataset_dir)
    assert result["record_count"] == 1
    records_path = Path(result["training_records_path"])
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line]
    assert rows[0]["targets"]["selected_mode_index"] == 0
    assert rows[0]["targets"]["thinking_mode_index"] == 0
    assert rows[0]["targets"]["prefix_profile_index"] == 0
    assert rows[0]["targets"]["response_budget_bucket_index"] == 2
    assert rows[0]["targets"]["reasoning_budget_bucket_index"] == 5
    assert rows[0]["targets"]["token_budget_bucket_index"] == 2


def test_build_training_record_rejects_invalid_bucket():
    dataset_row = {
        "export_key": "tr-1|req-1|dec-1|1700000000000",
        "transition": make_transition(response_budget_bucket=999),
    }
    try:
        build_training_record("vicuna-local-v1", dataset_row)
    except ValueError as exc:
        assert "unsupported response_budget_bucket" in str(exc)
    else:
        raise AssertionError("expected invalid token budget bucket to fail")
