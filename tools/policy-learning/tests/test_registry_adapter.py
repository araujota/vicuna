#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from policy_dataset import append_transitions
from policy_trainer import train_policy
from policy_training_contract import build_training_corpus


def test_registry_policy_adapter_replays_artifact(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    transition = {
        "transition_id": "tr-1",
        "request_id": "req-1",
        "decision_id": "dec-1",
        "created_at_ms": 1700000000000,
        "observation": {
            "mode_label": "chat",
            "bridge_scoped": False,
            "cognitive_replay": False,
            "heuristic_matched": False,
            "available_tool_count": 1,
            "parallel_tool_calls_requested": True,
            "input_message_count": 1,
            "ongoing_task_due": 0.0,
            "vad": {"valence": 0.0, "arousal": 0.2, "dominance": 0.4},
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
            "selected_mode": "tool_light",
            "reasoning_depth": "short",
            "token_budget_bucket": 512,
            "tool_parallelism_cap": 0,
            "interrupt_allowed": False,
            "replan_required": False,
            "early_stop_ok": True,
            "force_synthesis": False,
        },
        "reward_events": [],
        "reward_total": 1.0,
        "terminated": True,
        "termination_reason": "stop",
    }
    append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload={
            "behavior_policy_version": "control_surface_v2",
            "candidate_policy_version": None,
        },
        transitions=[transition],
    )
    build_training_corpus(dataset_dir)
    trained = train_policy(
        dataset_dir=dataset_dir,
        model_name="vicuna-governance",
        run_root=tmp_path / "runs",
    )

    adapter = TOOLS_ROOT / "registry_policy_adapter.py"
    request_payload = {
        "observation": transition["observation"],
        "action_mask": transition["action_mask"],
    }
    result = subprocess.run(
        [sys.executable, str(adapter), "--artifact", trained["artifact_path"]],
        input=json.dumps(request_payload),
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["policy_version"] == trained["policy_version"]
    assert payload["action"]["schema_version"] == "policy_action_v2"
    assert payload["action"]["selected_mode"] == "tool_light"
    assert payload["action"]["thinking_mode"] in {"enabled", "disabled"}
    assert payload["action"]["proposal_source"] == "registry_artifact"
    assert payload["confidence"]["overall"] > 0.0
    assert payload["confidence"]["feature_signature_seen"] is True
