#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from urllib import request


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from policy_dataset import append_transitions
from policy_registry import promote_alias, register_artifact
from policy_registry_server import create_server
from policy_trainer import train_policy
from policy_training_contract import build_training_corpus


def bootstrap_registry(tmp_path: Path) -> Path:
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
        transitions=[
            {
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
                    "vad": {"valence": 0.0, "arousal": 0.2, "dominance": 0.4},
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
                    "selected_mode": "tool_light",
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
                "reward_events": [],
                "reward_total": 1.0,
                "terminated": True,
                "termination_reason": "stop",
            }
        ],
    )
    build_training_corpus(dataset_dir)
    trained = train_policy(
        dataset_dir=dataset_dir,
        model_name="vicuna-governance",
        run_root=tmp_path / "runs",
    )
    evaluation_report_path = tmp_path / "offline_eval.json"
    evaluation_report_path.write_text(
        '{"exact_match_rate": 1.0, "invalid_action_rate": 0.0, "reward_total_mean_on_match": 1.0}\n',
        encoding="utf-8",
    )
    registry_dir = tmp_path / "registry"
    registered = register_artifact(
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        artifact_path=Path(trained["artifact_path"]),
        training_run_manifest_path=Path(trained["training_run_manifest_path"]),
        evaluation_report_path=evaluation_report_path,
        tags={"validation_status": "passed"},
    )
    promote_alias(
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        alias="candidate",
        version=registered["version"],
        reason="candidate rollout target",
    )
    return registry_dir


def _request_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    body = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = request.Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8"))


def test_registry_server_health_and_propose(tmp_path: Path, capsys):
    registry_dir = bootstrap_registry(tmp_path)
    server = create_server(
        host="127.0.0.1",
        port=0,
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        default_alias="candidate",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        health = _request_json(base_url, "/health")
        assert health["ok"] is True
        assert health["resolved_default_alias"] == "candidate"
        assert health["resolved_default_policy_version"].startswith("vicuna-governance-")

        response = _request_json(
            base_url,
            "/v1/policy/propose",
            payload={
                "policy_mode": "canary_live",
                "observation": {
                    "mode_label": "chat",
                    "bridge_scoped": False,
                    "cognitive_replay": False,
                    "heuristic_matched": False,
                    "available_tool_count": 1,
                    "parallel_tool_calls_requested": True,
                    "input_message_count": 1,
                    "vad": {"valence": 0.0, "arousal": 0.2, "dominance": 0.4},
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
            },
        )
        assert response["policy_alias"] == "candidate"
        assert response["policy_version"].startswith("vicuna-governance-")
        assert response["action"]["selected_mode"] == "tool_light"
        assert response["confidence"]["overall"] > 0.0
        assert response["confidence"]["feature_signature_seen"] is True

        resolved = _request_json(
            base_url,
            "/v1/artifacts/resolve",
            payload={
                "artifact_alias": "candidate",
            },
        )
        assert resolved["artifact_kind"] == "policy"
        assert resolved["artifact_alias"] == "candidate"
        assert resolved["artifact"]["schema_version"] == "vicuna.policy_artifact.v1"
        captured = capsys.readouterr()
        assert '"event": "health_checked"' in captured.out
        assert '"event": "proposal_served"' in captured.out
        assert '"event": "artifact_resolved"' in captured.out
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()
