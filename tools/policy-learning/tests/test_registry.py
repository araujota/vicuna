#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parent
TOOLS_ROOT = TESTS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from policy_dataset import append_transitions
from policy_registry import promote_alias, register_artifact, registry_status, resolve_artifact_path
from policy_trainer import train_policy
from policy_training_contract import build_training_corpus


def bootstrap_artifact(tmp_path: Path) -> tuple[Path, dict]:
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
                    "available_tool_count": 0,
                    "parallel_tool_calls_requested": False,
                    "input_message_count": 1,
                    "ongoing_task_due": 0.0,
                    "vad": {"valence": 0.0, "arousal": 0.2, "dominance": 0.4},
                },
                "action_mask": {
                    "allowed_modes": ["direct"],
                    "allowed_reasoning_depths": ["short"],
                    "max_tool_parallelism_cap": 0,
                    "allow_interrupt": False,
                    "allow_replan": False,
                    "allow_early_stop": True,
                    "allow_force_synthesis": False,
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
    return evaluation_report_path, trained


def test_registry_versions_and_aliases(tmp_path: Path):
    registry_dir = tmp_path / "registry"
    evaluation_report_path, trained = bootstrap_artifact(tmp_path)

    first = register_artifact(
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        artifact_path=Path(trained["artifact_path"]),
        training_run_manifest_path=Path(trained["training_run_manifest_path"]),
        evaluation_report_path=evaluation_report_path,
        tags={"validation_status": "passed"},
    )
    second = register_artifact(
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
        version=first["version"],
        reason="initial candidate",
    )
    promote_alias(
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        alias="candidate",
        version=second["version"],
        reason="new candidate",
    )

    status = registry_status(registry_dir, "vicuna-governance")
    assert status["version_count"] == 2
    assert status["aliases"]["candidate"] == 2
    assert len(status["promotion_history"]) == 2
    resolved = resolve_artifact_path(
        registry_dir=registry_dir,
        model_name="vicuna-governance",
        alias="candidate",
    )
    assert resolved.exists()
