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
from ppo_evaluator import evaluate_ppo_policy
from ppo_trainer import train_ppo_policy
from ppo_training_contract import build_ppo_training_corpus, load_ppo_training_records


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


def _transition(*, request_id: str, mode: str, reward_total: float, live: bool) -> dict:
    return {
        "transition_id": f"tr-{request_id}",
        "request_id": request_id,
        "decision_id": f"dec-{request_id}",
        "created_at_ms": 1700000000000,
        "candidate_executed_live": live,
        "policy_rollout": {
            "available": live,
            "artifact_kind": "ppo_policy",
            "policy_version": "seed-policy-v1",
            "selected_log_prob": -0.2,
            "value_estimate": reward_total - 0.1,
            "entropy": 0.4,
        },
        "observation": {
            "mode_label": "chat",
            "bridge_scoped": False,
            "cognitive_replay": False,
            "heuristic_matched": False,
            "available_tool_count": 1,
            "parallel_tool_calls_requested": True,
            "input_message_count": 1,
            "moment": {
                "confidence": 0.7 if mode == "direct" else 0.4,
                "curiosity": 0.2,
                "frustration": 0.1 if mode == "direct" else 0.3,
                "satisfaction": 0.6 if mode == "direct" else 0.2,
                "momentum": 0.5,
                "caution": 0.3,
                "stall": 0.1,
                "epistemic_pressure": 0.2 if mode == "direct" else 0.6,
                "planning_clarity": 0.8 if mode == "direct" else 0.4,
                "user_alignment": 0.7,
                "semantic_novelty": 0.5,
                "runtime_trust": 0.8,
                "runtime_failure_pressure": 0.1,
                "contradiction_pressure": 0.1,
            },
            "vad": {"valence": 0.2, "arousal": 0.1, "dominance": 0.3},
            "tool_context": {
                "available_tool_count": 1,
                "parallel_tool_calls_requested": True,
                "correctness": {
                    "available": True,
                    "score": 0.9,
                    "confidence": 0.8,
                },
            },
        },
        "action_mask": {
            "allowed_modes": ["direct", "tool_light"],
            "allowed_reasoning_depths": ["short", "medium"],
            "allowed_response_budget_buckets": [256, 512, 1024],
            "allowed_reasoning_budget_buckets": [0, 64, 128],
            "allowed_thinking_modes": ["enabled", "disabled"],
            "allowed_prefix_profiles": ["none", "bounded_answer"],
            "allowed_stop_profiles": ["none", "concise_answer"],
            "allowed_sampling_profiles": ["provider_default", "balanced"],
            "allowed_repetition_profiles": ["none", "anti_stall_soft"],
            "allowed_tool_choice_profiles": ["caller_default", "auto"],
            "max_tool_parallelism_cap": 1,
            "allow_interrupt": True,
            "allow_replan": True,
            "allow_early_stop": True,
            "allow_force_synthesis": True,
        },
        "executed_action": {
            "selected_mode": mode,
            "reasoning_depth": "short" if mode == "direct" else "medium",
            "thinking_mode": "enabled",
            "prefix_profile": "bounded_answer",
            "stop_profile": "concise_answer",
            "sampling_profile": "balanced" if mode == "tool_light" else "provider_default",
            "repetition_profile": "none",
            "tool_choice_profile": "caller_default",
            "response_budget_bucket": 512,
            "reasoning_budget_bucket": 64 if mode == "direct" else 128,
            "token_budget_bucket": 512,
            "tool_parallelism_cap": 0,
            "interrupt_allowed": False,
            "replan_required": False,
            "early_stop_ok": True,
            "force_synthesis": False,
        },
        "reward_events": [],
        "reward_total": reward_total,
        "terminated": True,
        "termination_reason": "stop",
    }


def _bootstrap_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    append_transitions(
        dataset_dir=dataset_dir,
        dataset_id="vicuna-ppo-local-v1",
        source_base_url="http://127.0.0.1:8080",
        export_mode="snapshot",
        status_payload={
            "behavior_policy_version": "control_surface_v3",
            "candidate_policy_version": "seed-policy-v1",
        },
        transitions=[
            _transition(request_id="1", mode="direct", reward_total=1.2, live=True),
            _transition(request_id="2", mode="tool_light", reward_total=0.4, live=False),
        ],
    )
    return dataset_dir


def test_build_train_and_serve_ppo_policy(tmp_path: Path) -> None:
    dataset_dir = _bootstrap_dataset(tmp_path)
    training_contract = build_ppo_training_corpus(dataset_dir)
    assert training_contract["record_count"] == 2
    manifest, records = load_ppo_training_records(dataset_dir)
    assert manifest["record_count"] == 2
    assert records[0]["rollout"]["available"] is True

    trained = train_ppo_policy(
        dataset_dir=dataset_dir,
        model_name="vicuna-governance",
        hidden_dims="16,16",
        warmstart_epochs=5,
        ppo_epochs=5,
        learning_rate=0.01,
        run_root=tmp_path / "runs",
    )
    assert trained["policy_version"].startswith("vicuna-governance-")

    report = evaluate_ppo_policy(
        dataset_dir=dataset_dir,
        artifact_path=Path(trained["artifact_path"]),
        report_path=tmp_path / "ppo_eval.json",
    )
    assert report["artifact_kind"] == "ppo_policy"
    assert report["exact_match_rate"] >= 0.0

    registry_dir = tmp_path / "registry"
    registered = register_artifact(
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        artifact_path=Path(trained["artifact_path"]),
        training_run_manifest_path=Path(trained["training_run_manifest_path"]),
        evaluation_report_path=Path(report["report_path"]),
        tags={"validation_status": "passed"},
    )
    assert registered["artifact_kind"] == "ppo_policy"
    promote_alias(
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        alias="candidate",
        version=registered["version"],
        reason="ppo candidate ready",
    )

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
        assert health["artifact_kind"] == "ppo_policy"

        response = _request_json(
            base_url,
            "/v1/policy/propose",
            payload={
                "policy_mode": "shadow",
                "observation": records[0]["observation"],
                "action_mask": records[0]["action_mask"],
            },
        )
        assert response["artifact_kind"] == "ppo_policy"
        assert response["rollout"]["available"] is True
        assert response["rollout"]["artifact_kind"] == "ppo_policy"
        assert response["confidence"]["overall"] > 0.0
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()
